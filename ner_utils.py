# -*- coding: utf-8 -*-
from collections import Counter, defaultdict

import numpy as np
import torch


def sample_fewshot_data(data, ratio, seed):
    '''从数据列表中随机采样指定比例的数据，保持原有顺序不变。'''
    if ratio >= 1.0:
        return data
    generator = torch.Generator()#固定随机数种子
    generator.manual_seed(seed)
    sample_size = max(1, int(len(data) * ratio))
    indices = torch.randperm(len(data), generator=generator)[:sample_size].tolist()#torch.randperm返回一个从 0 到 n-1 的整数随机排列（一维张量）。
    indices.sort()
    return [data[index] for index in indices]


def build_entity_frequency(data):
    '''Args:
        data: 数据列表，每条数据是一个列表，第一个元素是文本字符串，后面是若干个实体标注，每个标注是一个三元组 (start_index, end_index, label)，表示实体在文本中的起止位置和实体类型标签。
    Returns:
        Counter: 一个计数字典，键是实体文本，值是该实体在数据中出现的频次。'''
    frequency = Counter()
    for sample in data:
        text = sample[0]
        for start_index, end_index, _ in sample[1:]:
            entity_text = text[start_index:end_index + 1].strip()
            if entity_text:
                frequency[entity_text] += 1
    return frequency


def decode_entities_from_labels(labels, raw_text_list, offset_mappings, id2ent):
    """
    从真实标签张量中解码出实体标注（用于评估时构造 gold 标准）。

    Args:
        labels (torch.Tensor): 真实标签张量，形状为
            ``[batch_size, num_entity_types, seq_len, seq_len]``，one-hot 风格。
            其中 ``labels[b, l, i, j] == 1`` 表示第 ``b`` 个样本中存在一个
            第 ``l`` 类实体，span 为 token ``i`` 到 token ``j``。
        raw_text_list (list[str]): 原始文本列表，长度等于 batch_size。
        offset_mappings (list[list[tuple[int, int]]]): token 到字符的偏移映射列表，
            格式同 ``decode_entities_from_logits``。

    Returns:
        list[list[tuple]]: 解码结果，每个实体为四元组
            ``(entity_type, start_char, end_char, entity_text)``，含义同上。
    """
    decoded = []
    labels_np = labels.detach().cpu().numpy()
    for batch_index, text in enumerate(raw_text_list):
        sample_entities = []
        offset_mapping = offset_mappings[batch_index]
        for label_id, token_start, token_end in zip(*np.where(labels_np[batch_index] > 0)):
            if token_start >= len(offset_mapping) or token_end >= len(offset_mapping):
                continue
            start_span = offset_mapping[token_start]
            end_span = offset_mapping[token_end]
            if not start_span or not end_span or start_span == (0, 0) or end_span == (0, 0):
                continue
            start_char = start_span[0]
            end_char = end_span[1] - 1
            entity_text = text[start_char:end_char + 1]
            sample_entities.append((id2ent[label_id], start_char, end_char, entity_text))
        decoded.append(sample_entities)
    return decoded


def decode_entities_from_logits(logits, raw_text_list, offset_mappings, threshold, id2ent):
    """
    从模型推理输出的 logits 中解码出实体预测结果。

    Args:
        logits (torch.Tensor): 模型输出的 span 打分张量，形状为
            ``[batch_size, num_entity_types, seq_len, seq_len]``。
            其中 ``logits[b, l, i, j]`` 表示第 ``b`` 个样本中，第 ``l`` 类实体
            从 token ``i`` 开始到 token ``j`` 结束的 span 得分。
        raw_text_list (list[str]): 原始文本列表，长度等于 batch_size。
            每个元素是对应样本的原始字符串，用于切出实体文本。
        offset_mappings (list[list[tuple[int, int]]]): token 到字符的偏移映射列表，
            长度等于 batch_size。每个内层列表对应一个样本，其中每个 tuple 为
            ``(start_char, end_char)``，表示该 token 在原始文本中的字符区间
            （左闭右开）。
        threshold (float): 置信度阈值，仅当 span 得分大于该值时才视为有效实体预测。

    Returns:
        list[list[tuple]]: 每个样本解码出的实体列表，外层列表长度等于 batch_size。
            每个实体为一个四元组：
            ``(entity_type, start_char, end_char, entity_text)``，其中：
            - ``entity_type`` (str): 实体类型名称，如 ``"dis"``、``"sym"`` 等；
            - ``start_char`` (int): 实体在原始文本中的起始字符索引（含）；
            - ``end_char`` (int): 实体在原始文本中的结束字符索引（含）；
            - ``entity_text`` (str): 实体对应的原始文本片段。
    """
        
    decoded = []
    logits_np = logits.detach().cpu().numpy()
    for batch_index, text in enumerate(raw_text_list):
        sample_entities = []
        offset_mapping = offset_mappings[batch_index]
        scores = logits_np[batch_index]         #zip后结果(类型, 起点, 终点)...
        for label_id, token_start, token_end in zip(*np.where(scores > threshold)):#np.where 接收一个布尔矩阵，返回满足条件的元素的坐标。(维度一[],维度二[],维度三[])
            if token_start >= len(offset_mapping) or token_end >= len(offset_mapping):
                continue
            start_span = offset_mapping[token_start]
            end_span = offset_mapping[token_end]
            if not start_span or not end_span or start_span == (0, 0) or end_span == (0, 0):
                continue
            start_char = start_span[0]
            end_char = end_span[1] - 1
            entity_text = text[start_char:end_char + 1]
            sample_entities.append((id2ent[label_id], start_char, end_char, entity_text))
        decoded.append(sample_entities)
    return decoded


def update_metric_counter(metric_counter, pred_entities, true_entities, entity_frequency, rare_threshold, long_entity_threshold):
    pred_set = set(pred_entities)
    true_set = set(true_entities)
    metric_counter["overall_pred"] += len(pred_set)
    metric_counter["overall_true"] += len(true_set)
    metric_counter["overall_hit"] += len(pred_set & true_set)

    pred_boundary_set = {(start, end) for _, start, end, _ in pred_entities}
    true_boundary_set = {(start, end) for _, start, end, _ in true_entities}
    metric_counter["boundary_pred"] += len(pred_boundary_set)
    metric_counter["boundary_true"] += len(true_boundary_set)
    metric_counter["boundary_hit"] += len(pred_boundary_set & true_boundary_set)

    for label, _, _, _ in pred_entities:
        metric_counter["per_type"][label]["pred"] += 1
    for label, _, _, _ in true_entities:
        metric_counter["per_type"][label]["true"] += 1
    for label, _, _, _ in pred_set & true_set:
        metric_counter["per_type"][label]["hit"] += 1

    pred_rare = {entity for entity in pred_set if entity_frequency.get(entity[3], 0) <= rare_threshold}
    true_rare = {entity for entity in true_set if entity_frequency.get(entity[3], 0) <= rare_threshold}
    metric_counter["rare_pred"] += len(pred_rare)
    metric_counter["rare_true"] += len(true_rare)
    metric_counter["rare_hit"] += len(pred_rare & true_rare)

    pred_long = {entity for entity in pred_set if len(entity[3]) >= long_entity_threshold}
    true_long = {entity for entity in true_set if len(entity[3]) >= long_entity_threshold}
    metric_counter["long_pred"] += len(pred_long)
    metric_counter["long_true"] += len(true_long)
    metric_counter["long_hit"] += len(pred_long & true_long)


def compute_prf(hit_count, pred_count, true_count):
    precision = hit_count / pred_count if pred_count > 0 else 0.0
    recall = hit_count / true_count if true_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def summarize_metric_counter(metric_counter):
    summary = {}
    summary["overall"] = compute_prf(
        metric_counter["overall_hit"],
        metric_counter["overall_pred"],
        metric_counter["overall_true"],
    )
    summary["boundary"] = compute_prf(
        metric_counter["boundary_hit"],
        metric_counter["boundary_pred"],
        metric_counter["boundary_true"],
    )
    summary["rare"] = compute_prf(
        metric_counter["rare_hit"],
        metric_counter["rare_pred"],
        metric_counter["rare_true"],
    )
    summary["long"] = compute_prf(
        metric_counter["long_hit"],
        metric_counter["long_pred"],
        metric_counter["long_true"],
    )

    per_type = {}
    for label, counts in metric_counter["per_type"].items():
        per_type[label] = compute_prf(counts["hit"], counts["pred"], counts["true"])
    summary["per_type"] = per_type
    return summary


def create_metric_counter():
    return {
        "overall_hit": 0,
        "overall_pred": 0,
        "overall_true": 0,
        "boundary_hit": 0,
        "boundary_pred": 0,
        "boundary_true": 0,
        "rare_hit": 0,
        "rare_pred": 0,
        "rare_true": 0,
        "long_hit": 0,
        "long_pred": 0,
        "long_true": 0,
        "per_type": defaultdict(lambda: {"hit": 0, "pred": 0, "true": 0}),#默认字典,传入方法当访问这些值都时候返回对应值
    }
