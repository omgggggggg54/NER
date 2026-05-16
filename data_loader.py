# -*- coding: utf-8 -*-
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_sample(text, entities):
    """把一条样本统一成 [text, (start, end, label_id), ...] 的内部格式。"""
    sample = [text]
    sample.extend(entities)
    return sample


def parse_bio_spans(text, bio_label, ent2id, sample_name):
    """把字符级 BIO 序列解析成 span 标注。

    这里按 IMCS 的官方标注口径处理：
    - 每个中文字符/标点对应一个 BIO 标签
    - 只允许 B-xxx / I-xxx / O
    - 标签数和文本字符数必须严格一致
    - 不做任何自动修补；每个 BIO 标签都必须被严格转换成对应 span
    """
    labels = bio_label.split()
    if len(labels) != len(text):
        raise ValueError(
            f"{sample_name} 的 BIO_label 长度和句子长度不一致："
            f"标签数={len(labels)}，字符数={len(text)}，句子内容={text}"
        )

    entities = []
    current_label = None
    current_start = None

    for index, tag in enumerate(labels):
        if tag == "O":
            if current_label is not None:
                entities.append((current_start, index - 1, ent2id[current_label]))
                current_label = None
                current_start = None
            continue

        if "-" not in tag:
            raise ValueError(f"{sample_name} 出现非法 BIO 标签：{tag}")

        prefix, label_name = tag.split("-", 1)
        if label_name not in ent2id:
            raise ValueError(f"{sample_name} 出现未注册实体类型：{label_name}")

        if prefix == "B":
            if current_label is not None:
                entities.append((current_start, index - 1, ent2id[current_label]))
            current_label = label_name
            current_start = index
            continue

        if prefix == "I":
            # 这里必须严格保证 I-标签延续的是同一实体。
            # 一旦出现孤立 I 或类型断裂，就直接报错，避免把原 BIO 标注“修补”成别的 span。
            if current_label is None:
                raise ValueError(
                    f"{sample_name} 在位置 {index} 出现孤立 I 标签：{tag}。"
                    "IMCS 的 BIO 转 span 采用严格模式，不做自动修补。"
                )
            if current_label != label_name:
                raise ValueError(
                    f"{sample_name} 在位置 {index} 出现 I 标签类型断裂：前一个实体类型为 {current_label}，"
                    f"当前标签为 {tag}。IMCS 的 BIO 转 span 采用严格模式，不做自动修补。"
                )
            continue

        raise ValueError(f"{sample_name} 出现非法 BIO 前缀：{prefix}")

    if current_label is not None:
        entities.append((current_start, len(labels) - 1, ent2id[current_label]))

    return entities


def load_cmeee_data(path, ent2id):
    """读取 CMeEE-V2，保持原有样本格式不变。"""
    data = []
    with open(path, encoding="utf-8") as file:
        for sample in json.load(file):
            entities = []
            for entity in sample["entities"]:
                start_index = entity["start_idx"]
                end_index = entity["end_idx"]
                label_name = entity["type"]
                if start_index <= end_index:
                    entities.append((start_index, end_index, ent2id[label_name]))
            data.append(normalize_sample(sample["text"], entities))
    return data


def load_imcs_dialogue_bio_data(path, ent2id):
    """读取 IMCS-V2-NER，并把对话级样本展平成句级样本。

    这里严格按照当前约定：
    - 一句 turn 就是一条样本
    - 只用 dialogue 里的 sentence
    - 没有实体的句子也保留
    """
    data = []
    with open(path, encoding="utf-8") as file:
        raw_data = json.load(file)

    for dialogue_id, dialogue_sample in raw_data.items():
        for turn in dialogue_sample.get("dialogue", []):
            text = turn.get("sentence", "")
            sample_name = f"IMCS dialogue_id={dialogue_id}, sentence_id={turn.get('sentence_id', '')}"
            bio_label = turn.get("BIO_label", "")

            # 测试集没有 BIO_label，因此这里只允许训练/验证集走这个入口。
            if not bio_label:
                raise ValueError(f"{sample_name} 缺少 BIO_label，不能作为训练/验证数据加载。")

            entities = parse_bio_spans(text, bio_label, ent2id, sample_name)
            data.append(normalize_sample(text, entities))

    return data


def load_data(path, dataset_config, ent2id):
    """按数据集输入格式分派到对应加载器。"""
    input_format = dataset_config["input_format"]
    if input_format == "cmeee_json":
        return load_cmeee_data(path, ent2id)
    if input_format == "imcs_dialogue_bio":
        return load_imcs_dialogue_bio_data(path, ent2id)
    raise ValueError(
        f"当前活动数据集 {dataset_config['dataset_name']} 的输入格式 {input_format} 不受支持。"
    )


class EntDataset(Dataset):
    """统一的实体识别数据集。

    不管底层来自 CMeEE 还是 IMCS，
    只要前面已经适配成 [text, (start, end, label_id), ...]，
    这里的编码、打标、padding 流程就完全不需要再改。
    """

    def __init__(
        self,
        data,
        tokenizer,
        ent2id,
        max_len=256,
        istrain=True,
        use_entity_replace_aug=False,
        entity_replace_prob=0.3,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.ent2id = ent2id
        self.max_len = max_len
        self.istrain = istrain
        self.use_entity_replace_aug = use_entity_replace_aug
        self.entity_replace_prob = entity_replace_prob

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        """对单条样本做 tokenizer 编码，并保留字符到 token 的映射。"""
        if not self.istrain:
            return None

        text = item[0]
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=self.max_len,
            truncation=True,
        )
        offset_mapping = [
            tuple(span) if span is not None else (0, 0)
            for span in encoded["offset_mapping"]
        ]
        start_mapping = {span[0]: index for index, span in enumerate(offset_mapping) if span != (0, 0)}
        end_mapping = {span[1] - 1: index for index, span in enumerate(offset_mapping) if span != (0, 0)}
        input_ids = encoded["input_ids"]
        token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))
        attention_mask = encoded["attention_mask"]
        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, offset_mapping

    def build_batch_entity_pool(self, examples):
        """从当前 batch 中收集同类实体文本，供同类替换增强使用。"""
        entity_pool = {label_id: [] for label_id in self.ent2id.values()}
        for item in examples:
            text = item[0]
            for start_index, end_index, label_id in item[1:]:
                entity_text = text[start_index:end_index + 1]
                if entity_text:
                    entity_pool[label_id].append(entity_text)
        return entity_pool

    def replace_entity_in_item(self, item, entity_pool):
        """随机替换一个实体文段，并同步修正后续实体偏移。"""
        if (not self.use_entity_replace_aug) or random.random() > self.entity_replace_prob or len(item) <= 1:
            return item

        entity_indices = list(range(1, len(item)))
        random.shuffle(entity_indices)
        for entity_index in entity_indices:
            start_index, end_index, label_id = item[entity_index]
            has_overlap_entity = any(
                current_index != entity_index
                and not (current_end < start_index or current_start > end_index)
                for current_index, (current_start, current_end, _) in enumerate(item[1:], start=1)
            )
            if has_overlap_entity:
                continue

            old_text = item[0][start_index:end_index + 1]
            candidates = [candidate for candidate in entity_pool.get(label_id, []) if candidate and candidate != old_text]
            if not candidates:
                continue

            new_text = random.choice(candidates)
            delta = len(new_text) - len(old_text)
            augmented_text = item[0][:start_index] + new_text + item[0][end_index + 1:]
            augmented_entities = []
            for current_index, (current_start, current_end, current_label) in enumerate(item[1:], start=1):
                if current_index == entity_index:
                    augmented_entities.append((current_start, current_start + len(new_text) - 1, current_label))
                elif current_start > end_index:
                    augmented_entities.append((current_start + delta, current_end + delta, current_label))
                else:
                    augmented_entities.append((current_start, current_end, current_label))
            return [augmented_text] + augmented_entities

        return item

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode="post"):
        """把 batch 内不同长度的序列 padding 到统一长度。"""
        if length is None:
            length = np.max([np.shape(item)[:seq_dims] for item in inputs], axis=0)
        elif not hasattr(length, "__getitem__"):
            length = [length]

        slices = [np.s_[:length[index]] for index in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for item in inputs:
            item = item[slices]
            for index in range(seq_dims):
                if mode == "post":
                    pad_width[index] = (0, length[index] - np.shape(item)[index])
                elif mode == "pre":
                    pad_width[index] = (length[index] - np.shape(item)[index], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            outputs.append(np.pad(item, pad_width, "constant", constant_values=value))
        return np.array(outputs)

    def collate(self, examples):
        """把一个 batch 整理成模型可直接吃的张量。"""
        raw_text_list = []
        batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], []
        batch_offset_mappings = []
        entity_pool = self.build_batch_entity_pool(examples)

        for item in examples:
            item = self.replace_entity_in_item(item, entity_pool)
            encoded = self.encoder(item)
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, offset_mapping = encoded

            # 标签矩阵形状固定为 [实体类别数, seq_len, seq_len]。
            labels = np.zeros((len(self.ent2id), self.max_len, self.max_len), dtype=np.int64)
            for start_index, end_index, label in item[1:]:
                if start_index in start_mapping and end_index in end_mapping:
                    token_start = start_mapping[start_index]
                    token_end = end_mapping[end_index]
                    labels[label, token_start, token_end] = 1

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, : len(input_ids), : len(input_ids)])
            batch_offset_mappings.append(offset_mapping)

        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

        return (
            raw_text_list,
            batch_input_ids,
            batch_attention_mask,
            batch_segment_ids,
            batch_labels,
            batch_offset_mappings,
        )

    def __getitem__(self, index):
        """这里只返回原始样本，真正编码放到 collate 阶段。"""
        return self.data[index]
