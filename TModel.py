# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCNAlignment(nn.Module):
    '''MCN co-energy 思想的 token 对齐层。

    原 MCN 用 segmentation attention 和 detection attention 做 co-energy 最大化。
    这里迁移成：边界 token 注意力 与 span token 注意力 在 token 相似矩阵上对齐。
    '''

    def __init__(self, temperature=1.0, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, boundary_hidden, context_hidden, start_logits, end_logits, span_logits, attention_mask):
        '''计算边界分支和 span 分支的 co-energy。

        Args:
            boundary_hidden: [B, L, H]，边界 GRU 得到的 token 表示。
            context_hidden: [B, L, H]，GlobalPointer 使用的 token 表示。
            start_logits/end_logits: [B, L]，边界头输出。
            span_logits: [B, E, L, L]，GlobalPointer span 打分。
            attention_mask: [B, L]，有效 token mask。
        '''
        mask = attention_mask.bool()
        very_negative = -1e12

        # 边界注意力来自起点/终点分数，表示“哪些 token 像实体边界”。
        boundary_scores = (start_logits + end_logits) / max(self.temperature, self.eps)
        boundary_scores = boundary_scores.masked_fill(~mask, very_negative)
        boundary_attention = torch.softmax(boundary_scores, dim=-1)

        # span 注意力从四维 span 矩阵反推到 token，表示“哪些 token 被主任务认为属于实体区域”。
        span_scores = span_logits.detach().max(dim=1).values
        token_scores = torch.maximum(span_scores.max(dim=-1).values, span_scores.max(dim=-2).values)
        token_scores = (token_scores / max(self.temperature, self.eps)).masked_fill(~mask, very_negative)
        span_attention = torch.softmax(token_scores, dim=-1)

        boundary_norm = F.normalize(boundary_hidden, p=2, dim=-1)
        context_norm = F.normalize(context_hidden.detach(), p=2, dim=-1)
        token_similarity = torch.matmul(boundary_norm, context_norm.transpose(1, 2))
        token_similarity = (token_similarity + 1.0) * 0.5

        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        token_similarity = token_similarity.masked_fill(~pair_mask, 0.0)
        co_energy = torch.bmm(boundary_attention.unsqueeze(1), token_similarity)
        co_energy = torch.bmm(co_energy, span_attention.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return -torch.log(co_energy.clamp_min(self.eps)).mean()


class TSBECLBoundaryFusion(nn.Module):
    '''TSBECL 论文公式(15)-(19)的 GRU 边界表示融合层。

    两个独立 GRU 分别学习实体 head/tail 隐表示，再用三个可学习线性投影严格实现：
    H = W1 * H_h + W2 * H_t + W3 * H_context。
    '''

    def __init__(self, hidden_size, gru_layers=1, dropout=0.1):
        super().__init__()
        gru_hidden_size = hidden_size // 2
        gru_dropout = dropout if gru_layers > 1 else 0.0
        self.head_gru = nn.GRU(
            hidden_size,
            gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.tail_gru = nn.GRU(
            hidden_size,
            gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)
        self.head_fusion = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tail_fusion = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_fusion = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        head_hidden, _ = self.head_gru(hidden_states)
        tail_hidden, _ = self.tail_gru(hidden_states)
        start_logits = self.start_classifier(head_hidden).squeeze(-1)
        end_logits = self.end_classifier(tail_hidden).squeeze(-1)
        fused_hidden = (
            self.head_fusion(head_hidden)
            + self.tail_fusion(tail_hidden)
            + self.context_fusion(hidden_states)
        )
        boundary_hidden = 0.5 * (head_hidden + tail_hidden)
        return start_logits, end_logits, fused_hidden, boundary_hidden


class MetricsCalculator(object):
    '''计算训练阶段常用的样本级指标。'''

    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true, threshold=0.0):
        '''按样本级别统计 F1。

        这里直接在四维 span 打分张量上做阈值化，再和真实标签做逐元素比较。
        这种写法虽然不是最终论文汇报的严格实体级 F1，但它对训练阶段观察收敛趋势很有用。
        '''
        y_pred = torch.gt(y_pred, threshold).float()
        numerator = 2 * torch.sum(y_true * y_pred)
        denominator = torch.sum(y_true + y_pred)
        if denominator.item() == 0:
            return torch.tensor(1.0, device=y_pred.device)
        return numerator / denominator

    def get_sample_precision(self, y_pred, y_true, threshold=0.0):
        '''按样本级别统计 Precision。'''
        y_pred = torch.gt(y_pred, threshold).float()
        denominator = y_pred.sum()
        if denominator.item() == 0:
            return torch.tensor(1.0, device=y_pred.device)
        return torch.sum(y_pred[y_true == 1]) / denominator

    def get_sample_recall(self, y_pred, y_true, threshold=0.0):
        '''按样本级别统计 Recall，和样本级 F1、Precision 使用同一套阈值口径。'''
        y_pred = torch.gt(y_pred, threshold).float()
        denominator = y_true.sum()
        if denominator.item() == 0:
            return torch.tensor(1.0, device=y_pred.device)
        return torch.sum(y_pred[y_true == 1]) / denominator

    def get_evaluate_fpr(self, y_pred, y_true, threshold=0.0):
        '''按作者 GlobalPointer 常用写法统计验证集 F1、Precision、Recall。

        该指标直接在四维 span 标签张量上统计命中的坐标数：
        - y_pred > threshold 的位置视为预测实体 span
        - y_true > 0 的位置视为真实实体 span
        - 两者相乘后求和得到命中数
        '''
        y_pred = torch.gt(y_pred, threshold).float()
        y_true = torch.gt(y_true, 0).float()
        hit_count = torch.sum(y_true * y_pred)
        pred_count = torch.sum(y_pred)
        true_count = torch.sum(y_true)

        precision = hit_count / pred_count if pred_count.item() > 0 else torch.tensor(0.0, device=y_pred.device)
        recall = hit_count / true_count if true_count.item() > 0 else torch.tensor(0.0, device=y_pred.device)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall).item() > 0
            else torch.tensor(0.0, device=y_pred.device)
        )
        return f1, precision, recall


class GlobalPointer(nn.Module):
    '''当前项目的统一实体识别模型。

    主干结构固定是：
    MacBERT -> 可选边界头 -> GlobalPointer 主打分

    其中：
    - GlobalPointer 是唯一保留的主预测头
    - 训练与推理都统一走 span 主任务，不再保留历史失败分支
    '''

    def __init__(
        self,
        encoder,
        ent_type_size,
        inner_dim,
        hidden_size=768,
        RoPE=True,
        use_boundary_head=True,
        boundary_bias_scale=0.25,
        use_tsbecl_boundary_fusion=False,
        tsbecl_boundary_gru_layers=1,
        tsbecl_boundary_dropout=0.1,
        use_mcn_alignment=False,
        mcn_alignment_temperature=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE
        self.use_boundary_head = use_boundary_head
        self.boundary_bias_scale = boundary_bias_scale
        self.use_tsbecl_boundary_fusion = use_tsbecl_boundary_fusion
        self.use_mcn_alignment = use_mcn_alignment

        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        if use_boundary_head:
            if use_tsbecl_boundary_fusion:
                self.boundary_fusion = TSBECLBoundaryFusion(
                    self.hidden_size,
                    gru_layers=tsbecl_boundary_gru_layers,
                    dropout=tsbecl_boundary_dropout,
                )
            else:
                self.start_classifier = nn.Linear(self.hidden_size, 1)
                self.end_classifier = nn.Linear(self.hidden_size, 1)
        if use_mcn_alignment:
            self.mcn_alignment = MCNAlignment(temperature=mcn_alignment_temperature)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        '''生成 RoPE 所需的位置编码。'''
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)#[seq_len,1]
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)#div_term(i) = 10000^(-2i/d_model) in RoPE paper
        embeddings = position_ids * indices#[seq_len, output_dim//2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        return torch.reshape(embeddings, (batch_size, seq_len, output_dim))#(batch_size, seq_len, output_dim)

    def apply_rope(self, query_states, key_states, input_ids):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, input_ids.device)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        query_states_2 = torch.stack([-query_states[..., 1::2], query_states[..., ::2]], dim=-1).reshape(query_states.shape)
        key_states_2 = torch.stack([-key_states[..., 1::2], key_states[..., ::2]], dim=-1).reshape(key_states.shape)
        query_states = query_states * cos_pos + query_states_2 * sin_pos
        key_states = key_states * cos_pos + key_states_2 * sin_pos
        return query_states, key_states

    def compute_query_key_states(self, hidden_states, input_ids):
        '''把隐藏状态投影成 GlobalPointer 需要的起点/终点表示。'''
        batch_size, seq_len, _ = hidden_states.size()
        outputs = self.dense(hidden_states)#[B,S,entity_type*inner_dim*2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)#entity_type*[B,S,inner_dim*2]
        outputs = torch.stack(outputs, dim=-2)#[B,S,entity_type,entity_type,inner_dim*2]
        query_states, key_states = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]#[B,S,entity_type,inner_dim]
        if self.RoPE:
            query_states, key_states = self.apply_rope(query_states, key_states, input_ids)
        return query_states, key_states

    def compute_span_logits(self, hidden_states, input_ids):
        '''保持原版 GlobalPointer 的点积打分口径。'''
        query_states, key_states = self.compute_query_key_states(hidden_states, input_ids)#[B,S,entity_type,inner_dim]
        return torch.einsum("bmhd,bnhd->bhmn", query_states, key_states)#[B,entity_type,seq_len,seq_len]

    def apply_boundary_bias(self, logits, start_logits, end_logits):
        '''把边界头的起点/终点偏置加回 span 打分。

        这样做的目的不是替代 GlobalPointer，而是给“起点像不像实体起点、终点像不像实体终点”
        这类信息一个直接影响 span 打分的通道。
        '''
        start_scores = torch.sigmoid(start_logits).unsqueeze(1).unsqueeze(-1)
        end_scores = torch.sigmoid(end_logits).unsqueeze(1).unsqueeze(2)
        return logits + self.boundary_bias_scale * (start_scores + end_scores)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        '''模型前向主流程。

        这里直接收口为主链真实需要的三个输入，避免外部脚本继续传历史残留参数。
        '''
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        context_hidden = context_outputs[0]
        last_hidden_state = context_hidden

        start_logits, end_logits = None, None
        boundary_hidden = None
        if self.use_boundary_head:
            if self.use_tsbecl_boundary_fusion:
                start_logits, end_logits, last_hidden_state, boundary_hidden = self.boundary_fusion(last_hidden_state)
            else:
                start_logits = self.start_classifier(last_hidden_state).squeeze(-1)#[B,L]
                end_logits = self.end_classifier(last_hidden_state).squeeze(-1)#[B,L]
                boundary_hidden = last_hidden_state

        # 最终打分保持原版 GlobalPointer 点积口径，保证主预测头足够稳定。
        logits = self.compute_span_logits(last_hidden_state, input_ids)
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        if self.use_boundary_head:
            logits = self.apply_boundary_bias(logits, start_logits, end_logits)

        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12#参数 diagonal = -1 表示从主对角线下方第一条对角线开始保留
        logits = logits / (self.inner_dim ** 0.5)

        return {
            "logits": logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "boundary_hidden": boundary_hidden,
            "context_hidden": last_hidden_state,
            "encoder_hidden": context_hidden,
        }
