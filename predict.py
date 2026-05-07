# -*- coding: utf-8 -*-
import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from data_loader import ent2id
from ner_utils import decode_entities_from_logits
from paper_config import PAPER_EXPLICIT_SETTINGS, REPRO_DEFAULTS
from TModel import GlobalPointer


def parse_args():
    parser = argparse.ArgumentParser(description="边界感知 GlobalPointer 推理脚本")
    parser.add_argument("--model_path", default=PAPER_EXPLICIT_SETTINGS["model_path"], help="预训练模型目录，或 HuggingFace 模型名")
    parser.add_argument("--checkpoint_path", default="./outputs/ent_model.pth", help="训练好的模型权重")
    parser.add_argument("--test_path", default=r"./CMeEE-V2/CMeEE-V2_test.json", help="待预测数据路径")
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


def build_model(args, device):
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


def ner_predict(text, tokenizer, ner_model, device, args):
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
    token_type_ids = torch.tensor(encoded.get("token_type_ids", [0] * len(encoded["input_ids"]))).long().unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = ner_model(
            input_ids,
            attention_mask,
            token_type_ids,
        )

    entities = decode_entities_from_logits(outputs["logits"], [text], [offset_mapping], args.prediction_threshold)[0]
    return {
        "text": text,
        "entities": [
            {
                "start_idx": start_index,
                "end_idx": end_index,
                "type": label,
            }
            for label, start_index, end_index, _ in entities
        ],
    }


def main():
    args = parse_args()
    device = build_device(args.device)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = build_model(args, device)

    with open(args.test_path, encoding="utf-8") as file:
        test_data = json.load(file)

    results = []
    for sample in tqdm(test_data):
        results.append(ner_predict(sample["text"], tokenizer, model, device, args))

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
