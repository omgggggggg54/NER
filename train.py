# -*- coding: utf-8 -*-
import os
import random
import time
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

from data_loader import EntDataset, load_data
from ner_utils import (
    build_entity_frequency,
    create_metric_counter,
    decode_entities_from_labels,
    decode_entities_from_logits,
    sample_fewshot_data,
    summarize_metric_counter,
    update_metric_counter,
)
from paper_config import (
    ACTIVE_DATASET,
    ACTIVE_DATASET_CONFIG,
    PAPER_EXPLICIT_SETTINGS,
    PAPER_META,
    REPRO_DEFAULTS,
    build_label_mappings,
    ensure_dataset_supports_ner,
)
from TModel import GlobalPointer, MetricsCalculator


class FGM(object):
    '''对 encoder embedding 做单步快速扰动。'''

    def __init__(self, model, epsilon=1.0, emb_name="encoder.embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None or self.emb_name not in name:
                continue
            grad_norm = torch.norm(parameter.grad)
            if grad_norm.item() == 0 or not torch.isfinite(grad_norm):
                continue
            self.backup[name] = parameter.data.clone()
            parameter.data.add_(self.epsilon * parameter.grad / grad_norm)

    def restore(self):
        for name, parameter in self.model.named_parameters():
            if name in self.backup:
                parameter.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser(description="边界感知 GlobalPointer 训练脚本")
    parser.add_argument("--model_path", default=PAPER_EXPLICIT_SETTINGS["model_path"], help="预训练模型目录，或 HuggingFace 模型名")
    parser.add_argument("--train_path", default=ACTIVE_DATASET_CONFIG["train_path"], help="训练集路径")
    parser.add_argument("--eval_path", default=ACTIVE_DATASET_CONFIG["eval_path"], help="验证集路径")
    parser.add_argument("--output_path", default="./outputs/ent_model.pth", help="模型权重输出路径")
    parser.add_argument("--batch_size", type=int, default=REPRO_DEFAULTS["batch_size"], help="batch size")
    parser.add_argument("--epochs", type=int, default=REPRO_DEFAULTS["epochs"], help="训练轮数")
    parser.add_argument("--use_early_stopping", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_early_stopping"], help="是否启用早停")
    parser.add_argument("--early_stopping_patience", type=int, default=REPRO_DEFAULTS["early_stopping_patience"], help="早停耐心值")
    parser.add_argument("--early_stopping_min_delta", type=float, default=REPRO_DEFAULTS["early_stopping_min_delta"], help="早停最小提升幅度")
    parser.add_argument("--lr", type=float, default=REPRO_DEFAULTS["lr"], help="学习率")
    parser.add_argument("--optimizer", default=REPRO_DEFAULTS["optimizer"], choices=["Adam", "AdamW"], help="优化器")
    parser.add_argument("--num_workers", type=int, default=REPRO_DEFAULTS["num_workers"], help="DataLoader 进程数")
    parser.add_argument("--device", default="cuda:0", help="训练设备，例如 cuda:0 或 cpu")
    parser.add_argument("--max_len", type=int, default=REPRO_DEFAULTS["max_len"], help="最大序列长度")
    parser.add_argument("--hidden_size", type=int, default=REPRO_DEFAULTS["hidden_size"], help="隐藏层维度")
    parser.add_argument("--inner_dim", type=int, default=REPRO_DEFAULTS["inner_dim"], help="GlobalPointer 内部维度")
    parser.add_argument("--boundary_loss_weight", type=float, default=REPRO_DEFAULTS["boundary_loss_weight"], help="边界损失权重")
    parser.add_argument("--consistency_loss_weight", type=float, default=REPRO_DEFAULTS["consistency_loss_weight"], help="一致性损失权重")
    parser.add_argument("--boundary_bias_scale", type=float, default=REPRO_DEFAULTS["boundary_bias_scale"], help="边界偏置缩放系数")
    parser.add_argument("--use_tsbecl_boundary_fusion", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_tsbecl_boundary_fusion"], help="是否启用 TSBECL GRU head/tail 边界融合")
    parser.add_argument("--tsbecl_boundary_gru_layers", type=int, default=REPRO_DEFAULTS["tsbecl_boundary_gru_layers"], help="TSBECL 边界 GRU 层数")
    parser.add_argument("--tsbecl_boundary_dropout", type=float, default=REPRO_DEFAULTS["tsbecl_boundary_dropout"], help="TSBECL 边界 GRU dropout")
    parser.add_argument("--fewshot_ratio", type=float, default=REPRO_DEFAULTS["fewshot_ratio"], help="少样本采样比例")
    parser.add_argument("--fewshot_seed", type=int, default=REPRO_DEFAULTS["fewshot_seed"], help="少样本采样随机种子")
    parser.add_argument("--prediction_threshold", type=float, default=REPRO_DEFAULTS["prediction_threshold"], help="预测阈值")
    parser.add_argument("--rare_threshold", type=int, default=REPRO_DEFAULTS["rare_threshold"], help="稀有实体阈值")
    parser.add_argument("--long_entity_threshold", type=int, default=REPRO_DEFAULTS["long_entity_threshold"], help="长实体长度阈值")
    parser.add_argument("--entity_replace_prob", type=float, default=REPRO_DEFAULTS["entity_replace_prob"], help="同类实体替换增强概率")
    parser.add_argument("--seed", type=int, default=REPRO_DEFAULTS["seed"], help="随机种子")
    parser.add_argument("--num_folds", type=int, default=PAPER_EXPLICIT_SETTINGS["num_folds"], help="交叉验证折数")
    parser.add_argument("--adv_epsilon", type=float, default=REPRO_DEFAULTS["adv_epsilon"], help="对抗扰动半径")
    parser.add_argument("--use_kfold", action=argparse.BooleanOptionalAction, default=PAPER_EXPLICIT_SETTINGS["use_kfold"], help="是否启用 5 折交叉验证")
    parser.add_argument("--use_boundary_head", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_boundary_head"], help="是否启用边界头")
    parser.add_argument("--use_adversarial", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_adversarial"], help="是否启用 FGM 对抗训练")
    parser.add_argument("--use_entity_replace_aug", action=argparse.BooleanOptionalAction, default=REPRO_DEFAULTS["use_entity_replace_aug"], help="是否启用 batch 内同类实体替换增强")
    return parser.parse_args()


def build_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multilabel_categorical_crossentropy(y_pred, y_true):
    '''多标签损失以0为基准点'''
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12#0的位置维持原y_pred值
    y_pred_pos = y_pred - (1 - y_true) * 1e12#1的位置维持-y_pred值
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def span_loss(y_true, y_pred):#span方块loss即[B,E,L,L]
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.float().reshape(batch_size * ent_type_size, -1)#展平
    y_pred = y_pred.float().reshape(batch_size * ent_type_size, -1)
    return multilabel_categorical_crossentropy(y_pred, y_true)


def generate_boundary_labels(labels):
    '''
    Returns:
        start_labels:[B,L],[true,false列表]
        end_labels:[B,L],[true,false列表]
        '''
    start_labels = (labels.sum(dim=(1, 3)) > 0).float()#[B,L]每个Seq位置是否为起始位置
    end_labels = (labels.sum(dim=(1, 2)) > 0).float()#[B,L]每个Seq位置是否为结束位置
    return start_labels, end_labels


def calculate_boundary_loss(start_logits, end_logits, start_labels, end_labels, attention_mask):
    mask = attention_mask.float()
    start_loss = F.binary_cross_entropy_with_logits(start_logits, start_labels, reduction="none")
    end_loss = F.binary_cross_entropy_with_logits(end_logits, end_labels, reduction="none")#none不聚合，保留每个元素独立的损失	与输入相同 [B, L]
    normalizer = torch.clamp(mask.sum(), min=1.0)
    return ((start_loss + end_loss) * mask).sum() / normalizer#算平均损失


def calculate_consistency_loss(logits, start_logits, end_logits, attention_mask):
    mask = attention_mask.float()
    start_from_span = logits.max(dim=1).values.max(dim=-1).values#[B, L],[b, i] = 第 b 个样本中，token i 作为起点时，所有实体类型、所有终点中的最高 span 得分。
    end_from_span = logits.max(dim=1).values.max(dim=-2).values#[B, L],[b, j] = token j 作为终点时，所有类型、所有起点中的最佳 span 得分。
    start_target = torch.sigmoid(start_from_span.detach())
    end_target = torch.sigmoid(end_from_span.detach())
    start_pred = torch.sigmoid(start_logits)
    end_pred = torch.sigmoid(end_logits)
    normalizer = torch.clamp(mask.sum(), min=1.0)
    loss = ((start_pred - start_target) ** 2 + (end_pred - end_target) ** 2) * mask#用均方误差（MSE）衡量边界头和主任务的距离
    return loss.sum() / normalizer


def build_dataloader(data, tokenizer, ent2id, args, shuffle, is_train=False):
    dataset = EntDataset(
        data,
        tokenizer=tokenizer,
        ent2id=ent2id,
        max_len=args.max_len,
        istrain=True,
        use_entity_replace_aug=is_train and args.use_entity_replace_aug,
        entity_replace_prob=args.entity_replace_prob,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate,#DataLoader 在加载一个 batch 的数据时，会调用 dataset.collate_fn 来整理这些数据，使其适合模型输入
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def build_model(args, device, ent2id):
    encoder = BertModel.from_pretrained(args.model_path, use_safetensors=False)#默认769维
    model = GlobalPointer(
        encoder=encoder,
        ent_type_size=len(ent2id),#实体类型数量
        inner_dim=args.inner_dim,
        hidden_size=args.hidden_size,
        RoPE=True,
        use_boundary_head=args.use_boundary_head,
        boundary_bias_scale=args.boundary_bias_scale,
        use_tsbecl_boundary_fusion=args.use_tsbecl_boundary_fusion,
        tsbecl_boundary_gru_layers=args.tsbecl_boundary_gru_layers,
        tsbecl_boundary_dropout=args.tsbecl_boundary_dropout,
    ).to(device)
    return model


def build_adversarial_trainer(model, args):
    '''按配置返回 FGM 对抗训练器。'''
    if not args.use_adversarial:
        return None
    return FGM(model, epsilon=args.adv_epsilon)


def build_optimizer_and_scheduler(model, train_loader, args):
    '''使用统一学习率，失败的分层学习率和 warmup 不再进入主线。'''
    parameter_groups = [
        {
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": args.lr,
        }
    ]
    optimizer_class = torch.optim.AdamW if args.optimizer == "AdamW" else torch.optim.Adam
    optimizer = optimizer_class(parameter_groups)
    total_steps = max(args.epochs * len(train_loader), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def compute_total_batch_loss(model, input_ids, attention_mask, segment_ids, labels, args):
    '''统一封装主任务、边界头和一致性损失，方便正常训练和对抗训练复用同一套口径。'''
    outputs = model(
        input_ids,
        attention_mask,
        segment_ids,
    )

    logits = outputs["logits"]
    total_batch_loss = span_loss(labels.float(), logits.float())
    boundary_loss = logits.new_tensor(0.0)
    consistency_loss = logits.new_tensor(0.0)

    if args.use_boundary_head and outputs["start_logits"] is not None and outputs["end_logits"] is not None:
        start_labels, end_labels = generate_boundary_labels(labels)
        boundary_loss = calculate_boundary_loss(
            outputs["start_logits"],
            outputs["end_logits"],
            start_labels,
            end_labels,
            attention_mask,
        )
        consistency_loss = calculate_consistency_loss(
            logits,
            outputs["start_logits"],
            outputs["end_logits"],
            attention_mask,
        )
        total_batch_loss = (
            total_batch_loss
            + args.boundary_loss_weight * boundary_loss
            + args.consistency_loss_weight * consistency_loss
        )
    return total_batch_loss, outputs, boundary_loss, consistency_loss


def summarize_module_status(args):
    '''整理当前实验真正启用的模块状态。

    这份摘要只反映现在代码里真实存在、真实参与训练的模块，
    避免继续沿用已经删除的旧模块口径。
    '''
    return {
        "MacBERT": True,
        "边界头": args.use_boundary_head,
        "早停": args.use_early_stopping,
        "早停patience": args.early_stopping_patience if args.use_early_stopping else "off",
        "早停min_delta": args.early_stopping_min_delta if args.use_early_stopping else "off",
        "TSBECL边界融合": args.use_tsbecl_boundary_fusion,
        "TSBECL边界GRU层数": args.tsbecl_boundary_gru_layers if args.use_tsbecl_boundary_fusion else "off",
        "对抗训练": args.use_adversarial,
        "对抗模式": "fgm" if args.use_adversarial else "off",
        "同类实体替换增强": args.use_entity_replace_aug,
        "LR": args.lr,
    }


def split_kfold(data, num_folds, seed):
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    fold_sizes = [len(indices) // num_folds] * num_folds#每个折的样本数[样本总数//折数] * 折数
    for index in range(len(indices) % num_folds):
        fold_sizes[index] += 1#将剩余的样本分配到前面几个折中

    folds = []
    start_index = 0
    for fold_size in fold_sizes:
        end_index = start_index + fold_size
        folds.append(indices[start_index:end_index])
        start_index = end_index
    return folds#返回一个列表，包含每个折的样本索引


def build_fold_output_path(output_path, fold_index, use_kfold):
    if not use_kfold:
        return output_path
    root, ext = os.path.splitext(output_path)
    ext = ext if ext else ".pth"
    return f"{root}_fold{fold_index + 1}{ext}"


def build_run_id():
    '''返回时间戳'''
    return time.strftime("%Y%m%d_%H%M%S")


def build_run_record_paths(output_path, run_id):#output_path模型权重输出目录
    output_dir = os.path.dirname(output_path) or "."
    record_dir = os.path.join(output_dir, "run_records")
    os.makedirs(record_dir, exist_ok=True)
    json_path = os.path.join(record_dir, f"{run_id}.json")
    md_path = os.path.join(record_dir, f"{run_id}.md")
    return json_path, md_path


def convert_to_serializable(value):
    if isinstance(value, dict):
        return {key: convert_to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_run_record(args, run_id, summary, json_path, md_path):
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": convert_to_serializable(vars(args)),
        "summary": convert_to_serializable(summary),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=4, ensure_ascii=False)

    lines = [
        f"# 实验记录 {run_id}",
        "",
        "## 1. 运行参数",
        "",
    ]
    for key, value in sorted(record["args"].items(), key=lambda item: item[0]):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## 2. 最终结果",
        "",
    ])
    for key, value in record["summary"].items():
        lines.append(f"- `{key}`: `{value}`")

    with open(md_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def get_peak_memory_text(device):
    if device.type != "cuda":
        return "CPU 模式不统计显存"
    return f"{torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB"


def evaluate_model(model, eval_loader, entity_frequency, args, device, fold_index, epoch_index, id2ent):
    metrics = MetricsCalculator()
    metric_counter = create_metric_counter()
    total_eval_f1 = 0.0
    total_eval_precision = 0.0
    total_eval_recall = 0.0
    eval_start_time = time.time()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Fold {fold_index + 1 if fold_index is not None else 1} Validating"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, offset_mappings = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids,
                attention_mask,
                segment_ids,
            )
            logits = outputs["logits"]
            eval_f1, eval_precision, eval_recall = metrics.get_evaluate_fpr(
                logits,
                labels,
                args.prediction_threshold,
            )
            total_eval_f1 += eval_f1.item()
            total_eval_precision += eval_precision.item()
            total_eval_recall += eval_recall.item()

            pred_entities_batch = decode_entities_from_logits(
                logits,
                raw_text_list,
                offset_mappings,
                args.prediction_threshold,
                id2ent,
            )
            true_entities_batch = decode_entities_from_labels(
                labels,
                raw_text_list,
                offset_mappings,
                id2ent,
            )
            for pred_entities, true_entities in zip(pred_entities_batch, true_entities_batch):
                update_metric_counter(
                    metric_counter,
                    pred_entities,
                    true_entities,
                    entity_frequency,
                    args.rare_threshold,
                    args.long_entity_threshold,
                )

    summary = summarize_metric_counter(metric_counter)
    eval_time = time.time() - eval_start_time
    overall_precision, overall_recall, overall_f1 = summary["overall"]
    boundary_precision, boundary_recall, boundary_f1 = summary["boundary"]
    rare_precision, rare_recall, rare_f1 = summary["rare"]
    long_precision, long_recall, long_f1 = summary["long"]
    avg_eval_f1 = total_eval_f1 / max(len(eval_loader), 1)
    avg_eval_precision = total_eval_precision / max(len(eval_loader), 1)
    avg_eval_recall = total_eval_recall / max(len(eval_loader), 1)

    print(
        "Fold：{}\tEPOCH：{}\tEntity_F1:{:.4f}\tEntity_Precision:{:.4f}\tEntity_Recall:{:.4f}\tTensor_F1:{:.4f}\tTensor_Precision:{:.4f}\tTensor_Recall:{:.4f}\tBoundary_F1:{:.4f}\tRare_F1:{:.4f}\tLong_F1:{:.4f}\tEval_Time:{:.2f}s\tPeak_Memory:{}".format(
            fold_index + 1 if fold_index is not None else 1,
            epoch_index + 1,
            overall_f1,
            overall_precision,
            overall_recall,
            avg_eval_f1,
            avg_eval_precision,
            avg_eval_recall,
            boundary_f1,
            rare_f1,
            long_f1,
            eval_time,
            get_peak_memory_text(device),
        )
    )
    type_f1_text = " ".join(
        [f"{label}:{scores[2]:.4f}" for label, scores in sorted(summary["per_type"].items(), key=lambda item: item[0])]
    )
    if type_f1_text:
        print(f"各类别 F1：{type_f1_text}")
    print(
        "EVAL_F1:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}".format(
            avg_eval_f1,
            avg_eval_precision,
            avg_eval_recall,
        )
    )

    return {
        "overall_f1": overall_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "rare_precision": rare_precision,
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
        "long_precision": long_precision,
        "long_recall": long_recall,
        "long_f1": long_f1,
        "tensor_f1": avg_eval_f1,
        "tensor_precision": avg_eval_precision,
        "tensor_recall": avg_eval_recall,
        "per_type": summary["per_type"],
    }


def train_one_fold(train_data, eval_data, tokenizer, args, device, ent2id, id2ent, fold_index=None):
    '''Args:
        fold_index: 当前折索引，从 0 开始，如果不使用 kfold 则为 None
        train_data: 当前折的训练数据列表，每条数据是一个 dict，包含 "text" 和 "entities" 字段
        eval_data: 当前折的验证数据列表，格式同 train_data
        tokenizer: 分词器对象，用于文本编码
    '''
    metrics = MetricsCalculator()
    train_data = sample_fewshot_data(train_data, args.fewshot_ratio, args.fewshot_seed + (fold_index or 0))
    entity_frequency = build_entity_frequency(train_data)

    train_loader = build_dataloader(train_data, tokenizer, ent2id, args, shuffle=True, is_train=True)
    eval_loader = build_dataloader(eval_data, tokenizer, ent2id, args, shuffle=False, is_train=False)
    model = build_model(args, device, ent2id)
    adversarial_trainer = build_adversarial_trainer(model, args)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, args)
    save_path = build_fold_output_path(args.output_path, fold_index or 0, args.use_kfold)
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    best_eval_result = {
        "best_epoch": 0,
        "overall_f1": 0.0,
        "overall_precision": 0.0,
        "overall_recall": 0.0,
    }
    epochs_without_improvement = 0
    for epoch_index in range(args.epochs):
        total_loss = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_boundary_loss = 0.0
        total_consistency_loss = 0.0
        total_adversarial_loss = 0.0
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_index, batch in enumerate(train_loader):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, offset_mappings = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)#token_type_ids
            labels = labels.to(device)

            optimizer.zero_grad()
            total_batch_loss, outputs, boundary_loss, consistency_loss = compute_total_batch_loss(
                model,
                input_ids,
                attention_mask,
                segment_ids,
                labels,
                args,
            )
            logits = outputs["logits"]
            adversarial_loss = logits.new_tensor(0.0)

            if not torch.isfinite(total_batch_loss):
                raise RuntimeError(
                    f"训练损失出现非有限值: batch={batch_index + 1}, loss={total_batch_loss.item()}, "
                    f"logits_min={logits.min().item()}, logits_max={logits.max().item()}"
                )

            total_batch_loss.backward()
            if adversarial_trainer is not None:
                adversarial_trainer.attack()
                adversarial_loss, _, _, _ = compute_total_batch_loss(
                    model,
                    input_ids,
                    attention_mask,
                    segment_ids,
                    labels,
                    args,
                )
                adversarial_loss.backward()
                adversarial_trainer.restore()
            optimizer.step()
            scheduler.step()

            sample_f1 = metrics.get_sample_f1(logits.detach(), labels, args.prediction_threshold)
            sample_precision = metrics.get_sample_precision(logits.detach(), labels, args.prediction_threshold)
            sample_recall = metrics.get_sample_recall(logits.detach(), labels, args.prediction_threshold)
            total_loss += total_batch_loss.item()
            total_f1 += sample_f1.item()
            total_precision += sample_precision.item()
            total_recall += sample_recall.item()
            total_boundary_loss += boundary_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_adversarial_loss += adversarial_loss.item()

            if (batch_index + 1) % 10 == 0:
                avg_loss = total_loss / (batch_index + 1)
                avg_f1 = total_f1 / (batch_index + 1)
                avg_precision = total_precision / (batch_index + 1)
                avg_recall = total_recall / (batch_index + 1)
                avg_boundary_loss = total_boundary_loss / (batch_index + 1)
                avg_consistency_loss = total_consistency_loss / (batch_index + 1)
                avg_adversarial_loss = total_adversarial_loss / (batch_index + 1)
                print(
                    f"Fold {fold_index + 1 if fold_index is not None else 1}, "
                    f"Epoch {epoch_index + 1}/{args.epochs}, Batch {batch_index + 1}/{len(train_loader)}, "
                    f"Train Loss: {avg_loss:.4f}, Train Precision: {avg_precision:.4f}, "
                    f"Train Recall: {avg_recall:.4f}, Train F1: {avg_f1:.4f}, "
                    f"Boundary Loss: {avg_boundary_loss:.4f}, Consistency Loss: {avg_consistency_loss:.4f}, "
                    f"Adv Loss: {avg_adversarial_loss:.4f}"
                )

        eval_result = evaluate_model(
            model,
            eval_loader,
            entity_frequency,
            args,
            device,
            fold_index,
            epoch_index,
            id2ent,
        )

        f1_improved = eval_result["overall_f1"] > best_eval_result["overall_f1"] + args.early_stopping_min_delta
        if f1_improved:
            torch.save(model.state_dict(), save_path)
            best_eval_result = {
                "best_epoch": epoch_index + 1,
                "overall_f1": eval_result["overall_f1"],
                "overall_precision": eval_result["overall_precision"],
                "overall_recall": eval_result["overall_recall"],
            }
            epochs_without_improvement = 0
            print(
                f"Fold {fold_index + 1 if fold_index is not None else 1} model saved, "
                f"best epoch: {best_eval_result['best_epoch']}, "
                f"best Precision: {best_eval_result['overall_precision']:.4f}, "
                f"best Recall: {best_eval_result['overall_recall']:.4f}, "
                f"best F1: {best_eval_result['overall_f1']:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"Fold {fold_index + 1 if fold_index is not None else 1} early-stopping monitor: "
                f"{epochs_without_improvement}/{args.early_stopping_patience} epoch(s) without "
                f"Entity_F1 improvement > {args.early_stopping_min_delta}"
            )
            if args.use_early_stopping and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Fold {fold_index + 1 if fold_index is not None else 1} early stopped at epoch {epoch_index + 1}. "
                    f"Best epoch: {best_eval_result['best_epoch']}, best F1: {best_eval_result['overall_f1']:.4f}"
                )
                break
    return best_eval_result


def main():
    args = parse_args()
    dataset_config = ensure_dataset_supports_ner(ACTIVE_DATASET_CONFIG)
    ent2id, id2ent = build_label_mappings(dataset_config)
    set_seed(args.seed)
    device = build_device(args.device)
    run_id = build_run_id()
    record_json_path, record_md_path = build_run_record_paths(args.output_path, run_id)#返回日志路径

    print(f"论文来源：{PAPER_META['title']}")
    print(f"论文 PDF：{PAPER_META['pdf_path']}")
    print(f"当前活动数据集：{ACTIVE_DATASET}")
    print(f"当前实验编号：{run_id}")
    print(f"当前训练主干：{args.model_path}")
    print(f"当前数据集任务类型：{dataset_config['task_type']}")
    print(f"当前数据集标签数：{len(ent2id)}")
    print(f"当前 5 折交叉验证：{args.use_kfold}，折数 = {args.num_folds}")
    print(f"当前 batch_size：{args.batch_size}")
    print(f"当前 max_len：{args.max_len}")
    print(f"当前 epochs：{args.epochs}")
    print(f"当前 early_stopping：{args.use_early_stopping}")
    print(f"当前 early_stopping_patience：{args.early_stopping_patience}")
    print(f"当前 early_stopping_min_delta：{args.early_stopping_min_delta}")
    print(f"当前 optimizer：{args.optimizer}")
    print(f"当前 adversarial：{args.use_adversarial}")
    print("当前对抗模式：fgm")
    print(f"当前 TSBECL边界融合：{args.use_tsbecl_boundary_fusion}")
    print(f"当前 lr：{args.lr}")
    print(f"当前 use_entity_replace_aug：{args.use_entity_replace_aug}")
    print(f"当前 entity_replace_prob：{args.entity_replace_prob}")
    module_status = summarize_module_status(args)
    for module_name, enabled in module_status.items():
        print(f"当前模块 {module_name}：{enabled}")
    print(f"当前 fewshot_ratio：{args.fewshot_ratio}")
    print(f"当前实验记录 JSON：{record_json_path}")
    print(f"当前实验记录 Markdown：{record_md_path}")

    if args.use_tsbecl_boundary_fusion and not args.use_boundary_head:
        raise ValueError("--use_tsbecl_boundary_fusion 依赖 --use_boundary_head，请同时开启边界头。")

    print("开始加载 tokenizer...")
    start_time = time.time()
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path, do_lower_case=True)
    print(f"tokenizer 加载完成，耗时 {time.time() - start_time:.2f} 秒")

    if args.use_kfold:
        print(f"开始读取训练集：{args.train_path}")
        start_time = time.time()
        train_data = load_data(args.train_path, dataset_config, ent2id)
        print(f"训练集读取完成，共 {len(train_data)} 条，耗时 {time.time() - start_time:.2f} 秒")

        print(f"开始读取验证集：{args.eval_path}")
        start_time = time.time()
        eval_data = load_data(args.eval_path, dataset_config, ent2id) if args.eval_path and os.path.exists(args.eval_path) else []
        print(f"验证集读取完成，共 {len(eval_data)} 条，耗时 {time.time() - start_time:.2f} 秒")

        labeled_data = train_data + eval_data
        print(f"开始做 {args.num_folds} 折切分，总样本数 {len(labeled_data)}")
        start_time = time.time()
        folds = split_kfold(labeled_data, args.num_folds, args.seed)
        print(f"{args.num_folds} 折切分完成，耗时 {time.time() - start_time:.2f} 秒")

        fold_best_scores = []
        for fold_index, eval_indices in enumerate(folds):
            eval_index_set = set(eval_indices)#每个折依次当验证集
            fold_train_data = [sample for index, sample in enumerate(labeled_data) if index not in eval_index_set]
            fold_eval_data = [sample for index, sample in enumerate(labeled_data) if index in eval_index_set]
            print(f"\n========== Fold {fold_index + 1}/{args.num_folds} ==========")
            best_result = train_one_fold(fold_train_data, fold_eval_data, tokenizer, args, device, ent2id, id2ent, fold_index)
            fold_best_scores.append(best_result)

        avg_best_f1 = sum(item["overall_f1"] for item in fold_best_scores) / len(fold_best_scores)
        avg_best_precision = sum(item["overall_precision"] for item in fold_best_scores) / len(fold_best_scores)
        avg_best_recall = sum(item["overall_recall"] for item in fold_best_scores) / len(fold_best_scores)
        print(
            f"\n5 折训练完成，平均最佳 Precision：{avg_best_precision:.4f}，"
            f"平均最佳 Recall：{avg_best_recall:.4f}，平均最佳 F1：{avg_best_f1:.4f}"
        )
        summary = {
            "mode": "kfold",
            "num_folds": args.num_folds,
            "fold_best_scores": [
                {
                    "best_epoch": item["best_epoch"],
                    "best_precision": round(item["overall_precision"], 6),
                    "best_recall": round(item["overall_recall"], 6),
                    "best_f1": round(item["overall_f1"], 6),
                }
                for item in fold_best_scores
            ],
            "avg_best_precision": round(avg_best_precision, 6),
            "avg_best_recall": round(avg_best_recall, 6),
            "avg_best_f1": round(avg_best_f1, 6),
        }
    else:
        train_data = load_data(args.train_path, dataset_config, ent2id)
        eval_data = load_data(args.eval_path, dataset_config, ent2id)
        best_result = train_one_fold(train_data, eval_data, tokenizer, args, device, ent2id, id2ent)
        print(
            f"\n单次训练完成，最佳 Epoch：{best_result['best_epoch']}，"
            f"最佳 Precision：{best_result['overall_precision']:.4f}，"
            f"最佳 Recall：{best_result['overall_recall']:.4f}，"
            f"最佳 F1：{best_result['overall_f1']:.4f}"
        )
        summary = {
            "mode": "single",
            "best_epoch": best_result["best_epoch"],
            "best_precision": round(best_result["overall_precision"], 6),
            "best_recall": round(best_result["overall_recall"], 6),
            "best_f1": round(best_result["overall_f1"], 6),
        }

    save_run_record(args, run_id, summary, record_json_path, record_md_path)
    print("\n本次实验最终结果：")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"实验记录已保存到：{record_json_path}")
    print(f"实验摘要已保存到：{record_md_path}")


if __name__ == "__main__":
    main()
