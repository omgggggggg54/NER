# -*- coding: utf-8 -*-
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

# 实体类型到数字ID的映射（9类医疗实体）
ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
# 反向映射：数字ID -> 实体类型名称
id2ent = {}
for key, value in ent2id.items():
    id2ent[value] = key


def load_data(path):
    """
    从 JSON 文件加载标注数据，转换为内部格式。

    Args:
        path (str): JSON 文件路径，格式如 CMeEE 数据集：
            [{"text": "...", "entities": [{"start_idx": 0, "end_idx": 3, "type": "dis"}, ...]}, ...]

    Returns:
        list: 每个元素为 [text, (start, end, label_id), ...] 的列表。
              例如：["患者今日开始服用阿莫西林，出现皮疹。", (9, 12, 7), (16, 17, 2)]
    """
    data = []
    with open(path, encoding="utf-8") as file:
        for sample in json.load(file):
            # 每条数据第一个元素是原始文本
            data.append([sample["text"]])
            # 遍历该样本中的所有实体标注
            for entity in sample["entities"]:
                start_index, end_index, label = entity["start_idx"], entity["end_idx"], entity["type"]
                if start_index <= end_index:  # 有效区间
                    # 将实体类型转换为数字ID，并存储为三元组
                    data[-1].append((start_index, end_index, ent2id[label]))
    return data


class EntDataset(Dataset):
    """自定义数据集，负责文本编码、实体标签构建和批处理整理。"""

    def __init__(
        self,
        data,
        tokenizer,
        max_len=256,
        istrain=True,
        use_entity_replace_aug=False,
        entity_replace_prob=0.3,
    ):
        """
        Args:
            data (list): load_data 返回的原始数据列表。
            tokenizer: HuggingFace 的 tokenizer 对象。
            max_len (int): 最大序列长度（截断/填充目标长度）。
            istrain (bool): 是否为训练模式。若为 False，则 encoder() 返回 None（不编码）。
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.istrain = istrain
        self.use_entity_replace_aug = use_entity_replace_aug
        self.entity_replace_prob = entity_replace_prob

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        """
        对单个样本进行编码，返回编码后的各种映射和特征。

        Args:
            item (list): [text, (start, end, label), ...] 格式的样本。

        Returns:
            tuple: (text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, offset_mapping)
                
                - text: 原始文本
                - start_mapping: dict {字符起始位置: token索引}（仅有效token）
                - end_mapping: dict {字符结束位置: token索引}（字符结束位置 = token span的结尾字符索引-1）
                - input_ids: token id 列表
                - token_type_ids: 片段标记列表（默认全0）
                - attention_mask: 注意力掩码
                - offset_mapping: 每个token对应的字符区间列表 [(start_char, end_char), ...]
        """
        if not self.istrain:
            return None

        text = item[0]
        # 编码文本，同时返回字符偏移映射
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,#token 与原始文本字符位置的映射关系
            max_length=self.max_len,
            truncation=True,
        )
        # 将 None 的偏移替换为 (0,0)
        offset_mapping = [
            tuple(span) if span is not None else (0, 0)
            for span in encoded["offset_mapping"]#整理出来为每个token对应的字符区间列表 [(start_char, end_char), ...
        ]
        # 构建字符起始位置 -> token索引的映射
        start_mapping = {span[0]: index for index, span in enumerate(offset_mapping) if span != (0, 0)}
        # 构建字符结束位置 -> token索引的映射（结束字符位置为 span[1]-1）
        end_mapping = {span[1] - 1: index for index, span in enumerate(offset_mapping) if span != (0, 0)}
        input_ids = encoded["input_ids"]
        # 某些 tokenizer 可能不返回 token_type_ids，默认全0
        token_type_ids = encoded.get("token_type_ids", [0] * len(input_ids))
        attention_mask = encoded["attention_mask"]

        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, offset_mapping

    def build_batch_entity_pool(self, examples):
        '''从当前 batch 的标注里收集同类实体文本，E-11 只使用 batch 内实体，不引入外部词典。'''
        entity_pool = {label_id: [] for label_id in id2ent}
        for item in examples:
            text = item[0]
            for start_index, end_index, label_id in item[1:]:
                entity_text = text[start_index:end_index + 1]
                if entity_text:
                    entity_pool[label_id].append(entity_text)
        return entity_pool

    def replace_entity_in_item(self, item, entity_pool):
        '''随机替换一个实体文段，并同步更新后续实体偏移，保证 token 标签仍能对齐。'''
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
        """
        将多个序列填充到相同长度（NumPy 实现）。

        Args:
            inputs (list of np.ndarray): 待填充的序列列表。
            length (int or list): 目标长度。若为 None，则取各序列在 seq_dims 维上的最大值。
            value: 填充值。
            seq_dims (int): 序列维度数目（例如 1 表示一维序列，3 表示三维张量）。
            mode (str): "post" 表示在末尾填充，"pre" 表示在开头填充。

        Returns:
            np.ndarray: 填充后的数组，形状为 (len(inputs), 目标长度, ...)。
        """
        if length is None:
            # 自动计算目标长度：取列表中的元素在前 seq_dims 维度上的最大值
            length = np.max([np.shape(item)[:seq_dims] for item in inputs], axis=0)
        elif not hasattr(length, "__getitem__"):
            length = [length]  # 转换为列表

        # 构造切片对象，用于截取每个序列的前 length[i] 个元素
        slices = [np.s_[:length[index]] for index in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        # 初始化每个维度的填充宽度为 (0,0)
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for item in inputs:
            # 先截断到目标长度
            item = item[slices]
            # 计算每个维度需要填充的数量
            for index in range(seq_dims):
                if mode == "post":
                    pad_width[index] = (0, length[index] - np.shape(item)[index])#每一维上需要填充的维度
                elif mode == "pre":
                    pad_width[index] = (length[index] - np.shape(item)[index], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            # 执行填充
            outputs.append(np.pad(item, pad_width, "constant", constant_values=value))#np.pad(array, pad_width, mode, **kwargs)
                                                                                      #pad_width:((before_1, after_1), (before_2, after_2), ...)开头和末尾分别填充的数量
        return np.array(outputs)

    def collate(self, examples):
        """
        将一个 batch 的样本整理成张量形式（用于 DataLoader 的 collate_fn）。

        Args:
            examples (list): 每个元素是 __getitem__ 返回的原始样本（即 [text, entities...]）。

        Returns:
            tuple:(
                raw_text_list,           # list of str
                batch_input_ids,         # torch.LongTensor, shape (batch_size, max_len)
                batch_attention_mask,    # torch.FloatTensor, shape (batch_size, max_len)
                batch_segment_ids,       # torch.LongTensor, shape (batch_size, max_len)
                batch_labels,            # torch.LongTensor, shape (batch_size, num_types, max_len, max_len)
                batch_offset_mappings    # list of offset_mapping
            )
        """
        raw_text_list = []
        batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], []
        batch_offset_mappings = []
        entity_pool = self.build_batch_entity_pool(examples)

        for item in examples:
            item = self.replace_entity_in_item(item, entity_pool)
            # 编码当前样本
            encoded = self.encoder(item)
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, offset_mapping = encoded

            # 初始化标签矩阵：实体类型数 x max_len x max_len，token级别的标签矩阵，标记实体的起始和结束位置关系
            labels = np.zeros((len(ent2id), self.max_len, self.max_len), dtype=np.int64)

            # 遍历样本中的所有实体标注（从 item[1:] 开始）
            for start_index, end_index, label in item[1:]:
                # 如果字符起始/结束位置能映射到 token 位置，则打标
                if start_index in start_mapping and end_index in end_mapping:
                    token_start = start_mapping[start_index]
                    token_end = end_mapping[end_index]
                    labels[label, token_start, token_end] = 1#标注数据

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            # 只保留有效长度部分（截取到当前样本的 input_ids 长度）
            batch_labels.append(labels[:, : len(input_ids), : len(input_ids)])
            batch_offset_mappings.append(offset_mapping)

        # 对 batch 内的所有序列进行填充，使其长度一致
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        # labels 是四维：[batch, num_types, seq_len, seq_len]，所以 seq_dims=3
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
        """
        返回原始样本（不在这里进行编码，编码放在 collate 中以支持多进程）。
        """
        return self.data[index]
