# -*- coding: utf-8 -*-
import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from paper_config import (
    ACTIVE_DATASET,
    ACTIVE_DATASET_CONFIG,
    PAPER_EXPLICIT_SETTINGS,
    REPRO_DEFAULTS,
    build_label_mappings,
    ensure_dataset_supports_ner,
)
from TModel import GlobalPointer


def parse_args():
    parser = argparse.ArgumentParser(description="边界感知 GlobalPointer 推理脚本")
    parser.add_argument("--model_path", default=PAPER_EXPLICIT_SETTINGS["model_path"], help="预训练模型目录，或 HuggingFace 模型名")
    parser.add_argument("--checkpoint_path", default="./outputs/ent_model.pth", help="训练好的模型权重")
    parser.add_argument("--test_path", default=ACTIVE_DATASET_CONFIG["test_path"], help="待预测数据路径")
    parser.add_argument("--output_path", default="./outputs/predict.json", help="预测结果输出路径")
    parser.add_argument("--max_len", type=int, default=REPRO_DEFAULTS["max_len"], help="最大长度")
    parser.add_argument("--hidden_size", type=int, default=REPRO_DEFAULTS["hidden_size"], help="隐藏层维度")
    parser.add_argument("--inner_dim", type=int, default=REPRO_DEFAULTS["inner_dim"], help="GlobalPointer 内部维度")
    parser.add_argument("--boundary_bias_scale", type=float, default=REPRO_DEFAULTS["boundary_bias_scale"], help="边界偏置缩放系数")
    parser.add_argument("--prediction_threshold", type=float, default=REPRO_DEFAULTS["prediction_threshold"], help="预测阈值")
    parser.add_argument("--use_boundary_head", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_boundary_head"], help="是否启用边界头")
    parser.add_argument("--use_tsbecl_boundary_fusion", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_tsbecl_boundary_fusion"], help="是否启用 TSBECL GRU head/tail 边界融合")
    parser.add_argument("--tsbecl_boundary_gru_layers", type=int, default=REPRO_DEFAULTS["tsbecl_boundary_gru_layers"], help="TSBECL 边界 GRU 层数")
    parser.add_argument("--tsbecl_boundary_dropout", type=float, default=REPRO_DEFAULTS["tsbecl_boundary_dropout"], help="TSBECL 边界 GRU dropout")
    parser.add_argument("--device", default="cuda:0", help="推理设备，例如 cuda:0 或 cpu")
    return parser.parse_args()


def build_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(args, device, ent2id):
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    encoder = BertModel.from_pretrained(args.model_path, use_safetensors=False)
    model = GlobalPointer(
        encoder=encoder,
        ent_type_size=len(ent2id),
        inner_dim=args.inner_dim,
        hidden_size=args.hidden_size,
        RoPE=True,
        use_boundary_head=args.use_boundary_head,
        boundary_bias_scale=args.boundary_bias_scale,
        use_tsbecl_boundary_fusion=args.use_tsbecl_boundary_fusion,
        tsbecl_boundary_gru_layers=args.tsbecl_boundary_gru_layers,
        tsbecl_boundary_dropout=args.tsbecl_boundary_dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
    model.eval()
    return tokenizer, model


def encode_text(text, tokenizer, device, args):
    """对单句文本做编码，并保留 offset_mapping 方便 span 回译。"""
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=args.max_len,
        truncation=True,
    )
    offset_mapping = [
        tuple(span) if span is not None else (0, 0)
        for span in encoded["offset_mapping"]
    ]
    input_ids = torch.tensor(encoded["input_ids"]).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        encoded.get("token_type_ids", [0] * len(encoded["input_ids"]))
    ).long().unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).float().unsqueeze(0).to(device)
    return input_ids, token_type_ids, attention_mask, offset_mapping


def predict_spans(text, tokenizer, ner_model, device, args, id2ent):
    """返回单句文本的原始 span 预测结果，并保留分数供后处理使用。"""
    input_ids, token_type_ids, attention_mask, offset_mapping = encode_text(text, tokenizer, device, args)
    with torch.no_grad():
        outputs = ner_model(
            input_ids,
            attention_mask,
            token_type_ids,
        )

    scores = outputs["logits"][0].detach().cpu().numpy()
    entities = []
    for label_id, token_start, token_end in zip(*np.where(scores > args.prediction_threshold)):
        if token_start >= len(offset_mapping) or token_end >= len(offset_mapping):
            continue
        start_span = offset_mapping[token_start]
        end_span = offset_mapping[token_end]
        if not start_span or not end_span or start_span == (0, 0) or end_span == (0, 0):
            continue
        start_char = start_span[0]
        end_char = end_span[1] - 1
        if start_char > end_char or end_char >= len(text):
            continue
        entities.append(
            {
                "type": id2ent[label_id],
                "start_idx": start_char,
                "end_idx": end_char,
                "text": text[start_char:end_char + 1],
                "score": float(scores[label_id, token_start, token_end]),
            }
        )
    return entities


def build_cmeee_result(text, entities):
    """把预测 span 组装成 CMeEE 提交格式。"""
    return {
        "text": text,
        "entities": [
            {
                "start_idx": entity["start_idx"],
                "end_idx": entity["end_idx"],
                "type": entity["type"],
            }
            for entity in entities
        ],
    }


def select_non_overlapping_entities(entities):
    """按分数从高到低贪心保留不重叠实体。

    IMCS 的提交格式是一条 BIO 序列，天然不允许重叠。
    这里固定采用我们事先约定的策略，避免引入额外开关。
    """
    sorted_entities = sorted(
        entities,
        key=lambda item: (-item["score"], item["start_idx"], item["end_idx"]),
    )
    selected_entities = []
    occupied_positions = set()
    for entity in sorted_entities:
        entity_positions = set(range(entity["start_idx"], entity["end_idx"] + 1))
        if entity_positions & occupied_positions:
            continue
        selected_entities.append(entity)
        occupied_positions.update(entity_positions)
    return sorted(selected_entities, key=lambda item: (item["start_idx"], item["end_idx"]))


def convert_entities_to_bio(text, entities):
    """把不重叠 span 转成字符级 BIO 序列。"""
    bio_labels = ["O"] * len(text)
    for entity in entities:
        start_index = entity["start_idx"]
        end_index = entity["end_idx"]
        entity_type = entity["type"]
        bio_labels[start_index] = f"B-{entity_type}"
        for index in range(start_index + 1, end_index + 1):
            bio_labels[index] = f"I-{entity_type}"
    return " ".join(bio_labels)


def predict_cmeee(test_data, tokenizer, model, device, args, id2ent):
    """按 CMeEE 的原提交格式输出。"""
    results = []
    for sample in tqdm(test_data):
        text = sample["text"]
        entities = predict_spans(text, tokenizer, model, device, args, id2ent)
        results.append(build_cmeee_result(text, entities))
    return results


def predict_imcs(test_data, tokenizer, model, device, args, id2ent):
    """按 IMCS-V2-NER 官方提交格式输出。

    输出结构为：
    {
        "dialogue_id": {
            "sentence_id": "BIO BIO BIO ..."
        }
    }
    """
    results = {}
    for dialogue_id, dialogue_sample in tqdm(test_data.items()):
        results[dialogue_id] = {}
        for turn in dialogue_sample.get("dialogue", []):
            sentence_id = str(turn["sentence_id"])
            text = turn.get("sentence", "")
            entities = predict_spans(text, tokenizer, model, device, args, id2ent)
            entities = select_non_overlapping_entities(entities)
            results[dialogue_id][sentence_id] = convert_entities_to_bio(text, entities)
    return results


def main():
    args = parse_args()
    dataset_config = ensure_dataset_supports_ner(ACTIVE_DATASET_CONFIG)
    ent2id, id2ent = build_label_mappings(dataset_config)
    device = build_device(args.device)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"当前活动数据集：{ACTIVE_DATASET}")
    print(f"当前数据集任务类型：{dataset_config['task_type']}")
    print(f"当前预测输入：{args.test_path}")
    print(f"当前预测输出格式：{dataset_config['predict_output_format']}")
    print(f"当前标签数：{len(ent2id)}")

    tokenizer, model = build_model(args, device, ent2id)

    with open(args.test_path, encoding="utf-8") as file:
        test_data = json.load(file)

    if dataset_config["predict_output_format"] == "cmeee_entities":
        results = predict_cmeee(test_data, tokenizer, model, device, args, id2ent)
    elif dataset_config["predict_output_format"] == "imcs_bio_dict":
        results = predict_imcs(test_data, tokenizer, model, device, args, id2ent)
    else:
        raise ValueError(
            f"当前活动数据集 {dataset_config['dataset_name']} 的预测输出格式 "
            f"{dataset_config['predict_output_format']} 不受支持。"
        )

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
