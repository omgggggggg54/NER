# -*- coding: utf-8 -*-

PAPER_META = {
    # 论文标题，用于实验日志里标明当前复现/对比的论文来源。
    "title": "Chinese medical named entity recognition based on multimodal information fusion and hybrid attention mechanism",
    # 本地论文 PDF 路径，只用于日志展示，不参与训练。
    "pdf_path": r"D:\learning\Article\journal.pone.0325660.pdf",
}


PAPER_EXPLICIT_SETTINGS = {
    # 下面这些是论文正文里明确写出来的设置
    # 预训练模型路径；当前使用本地 MacBERT-base 中文权重。
    "model_path": "./models/chinese-macbert-base",
    # 是否默认使用 K 折交叉验证；True 表示训练时默认跑多折。
    "use_kfold": True,
    # K 折数量；论文口径默认 5 折。
    "num_folds": 5,
}

REPRO_DEFAULTS = {
    # 下面这些数值型超参数，论文正文没有公开写死
    # 这里给的是当前主链真实使用的默认实验配置。
    # tokenizer 截断/补齐后的最大 token 长度。
    "max_len": 256,
    # 单张卡/单进程下 DataLoader 每个 batch 的样本数。
    "batch_size": 16,
    # 每一折或单次训练的 epoch 数。
    "epochs": 15,
    # 是否启用早停；按验证集实体级 overall_f1 判断是否停止。
    "use_early_stopping": True,
    # 早停耐心值；连续多少个 epoch 没有有效提升就停止当前折训练。
    "early_stopping_patience": 3,
    # 早停最小提升幅度；提升超过该值才算有效变好。
    "early_stopping_min_delta": 1e-4,
    # 统一学习率；关闭分层学习率时使用这个值。
    "lr": 2e-5,
    # 优化器类型；训练脚本支持 Adam 和 AdamW。
    "optimizer": "AdamW",
    # DataLoader 工作进程数；Windows 下过大可能增加启动开销。
    "num_workers": 8,
    # 模型隐藏层维度；MacBERT-base 默认为 768。
    "hidden_size": 768,
    # GlobalPointer 内部 query/key 向量维度。
    "inner_dim": 64,
    # 是否启用实体起点/终点边界辅助头；仅 GlobalPointer 主线使用。
    "use_boundary_head": True,
    # 边界头 BCE 损失权重；仅 use_boundary_head=True 时有效。
    "boundary_loss_weight": 1.0,
    # span 主任务和边界头一致性损失权重；仅边界头开启时有效。
    "consistency_loss_weight": 0.2,
    # 边界头预测分数加回 span logits 时的缩放系数。
    "boundary_bias_scale": 0.25,
    # 是否启用 TSBECL 公式(15)-(19)的 GRU head/tail 边界融合。
    "use_tsbecl_boundary_fusion": True,
    # TSBECL 边界 GRU 层数。
    "tsbecl_boundary_gru_layers": 3,
    # TSBECL 边界 GRU dropout；仅多层 GRU 时实际生效。
    "tsbecl_boundary_dropout": 0.1,
    # few-shot 采样比例；1.0 表示使用完整训练集。
    "fewshot_ratio": 1.0,
    # few-shot 采样随机种子，保证子集可复现。
    "fewshot_seed": 42,
    # 解码阈值；GlobalPointer logits 大于该值才认为是预测实体。
    "prediction_threshold": 0.0,
    # 稀有实体阈值；训练集中出现次数 <= 该值会被计入 Rare_F1。
    "rare_threshold": 3,
    # 长实体长度阈值；实体文本长度 >= 该值会被计入 Long_F1。
    "long_entity_threshold": 4,
    # 是否启用 FGM 对抗训练。
    "use_adversarial": True,
    # 对抗扰动半径，控制 embedding 扰动强度。
    "adv_epsilon": 1.0,
    # 是否启用 batch 内同类实体替换增强；只作用于训练集。
    "use_entity_replace_aug": True,
    # E-11 每条训练样本触发同类实体替换的概率。
    "entity_replace_prob": 0.3,
    # 全局随机种子，用于 random/numpy/torch 复现。
    "seed": 42,
}
 
