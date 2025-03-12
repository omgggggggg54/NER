# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


# 门控图神经网络（GGNN）
class GGNN(nn.Module):
    def __init__(self, hidden_size, num_edge_types, num_steps=3):
        super(GGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.num_steps = num_steps

        # 为每种边类型定义一个权重矩阵
        self.edge_weights = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_edge_types)
        ])

        # GRU单元
        self.gru = nn.GRUCell(hidden_size, hidden_size)

        # 输出层
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, node_features, edge_matrix):
        """
        node_features: (batch_size, num_nodes, hidden_size)
        edge_matrix: (batch_size, num_edge_types, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = node_features.size()
        device = node_features.device

        # 初始隐藏状态
        h = node_features

        for step in range(self.num_steps):
            # 消息传递
            m = torch.zeros(batch_size, num_nodes, self.hidden_size).to(device)

            for edge_type in range(self.num_edge_types):
                # (batch_size, num_nodes, num_nodes) @ (batch_size, num_nodes, hidden_size)
                adj = edge_matrix[:, edge_type, :, :]
                m_edge = torch.bmm(adj, self.edge_weights[edge_type](h))
                m = m + m_edge

            # 更新节点表示
            h_reshaped = h.view(-1, self.hidden_size)
            m_reshaped = m.view(-1, self.hidden_size)

            h_new = self.gru(m_reshaped, h_reshaped)
            h = h_new.view(batch_size, num_nodes, self.hidden_size)

        # 输出层
        output = self.out_layer(torch.cat([node_features, h], dim=-1))

        return output


# 条件随机场（CRF）
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # 规定从START到各个状态的转移，以及从各个状态到END的转移
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, masks, tags=None):
        """计算CRF得分或损失

        Args:
            emissions: (batch_size, seq_len, num_tags)
            masks: (batch_size, seq_len)
            tags: (batch_size, seq_len)

        Returns:
            scores或loss
        """
        if tags is not None:
            return self._calculate_loss(emissions, masks, tags)
        else:
            return self._decode(emissions, masks)

    def _calculate_loss(self, emissions, masks, tags):
        """计算负对数似然
        """
        batch_size, seq_len, _ = emissions.size()

        # 计算所有可能序列的分数（归一化因子）
        scores = self._forward_alg(emissions, masks)

        # 计算真实序列的分数
        gold_scores = self._score_sentence(emissions, tags, masks)

        # 负对数似然：正确路径的分数 - 所有路径的分数
        return torch.mean(scores - gold_scores)

    def _forward_alg(self, emissions, masks):
        """计算归一化因子（所有可能序列的分数）
        """
        batch_size, seq_len, num_tags = emissions.size()

        # 初始化
        alpha = self.start_transitions.unsqueeze(0).expand(batch_size, num_tags)

        for i in range(seq_len):
            emit_score = emissions[:, i].unsqueeze(2)  # (batch_size, num_tags, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)

            # (batch_size, num_tags, 1) + (1, num_tags, num_tags) + (batch_size, 1, num_tags)
            # = (batch_size, num_tags, num_tags)
            next_score = emit_score + trans_score + alpha.unsqueeze(2)

            # 对所有可能的上一个状态做log_sum_exp, 得到当前状态的分数
            next_score = torch.logsumexp(next_score, dim=1)

            # 根据mask更新alpha
            mask = masks[:, i].unsqueeze(1)
            alpha = torch.where(mask.bool(), next_score, alpha)

        # 加上转移到END的分数
        alpha = alpha + self.end_transitions.unsqueeze(0)

        # 计算所有路径的分数
        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, emissions, tags, masks):
        """计算给定序列的分数
        """
        batch_size, seq_len, _ = emissions.size()
        score = self.start_transitions[tags[:, 0]]

        for i in range(seq_len - 1):
            # 发射分数
            emit_score = emissions[torch.arange(batch_size), i, tags[:, i]]

            # 转移分数
            trans_score = self.transitions[tags[:, i], tags[:, i + 1]]

            # 当前步的分数 = 发射分数 + 转移分数
            score = score + emit_score * masks[:, i] + trans_score * masks[:, i + 1]

        # 最后一个标签的发射分数
        last_ix = torch.sum(masks, dim=1) - 1
        last_ix[last_ix < 0] = 0
        score = score + emissions[torch.arange(batch_size), last_ix.long(), tags[:, -1]] * masks[:, -1]

        # 转移到END的分数
        score = score + self.end_transitions[tags[:, -1]] * masks[:, -1]

        return score

    def _decode(self, emissions, masks):
        """维特比解码，找出最可能的标签序列
        """
        batch_size, seq_len, _ = emissions.size()

        # 初始化
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            # 广播当前分数以计算所有可能的转移
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)

            # 计算所有可能的新分数
            broadcast_emission = emissions[:, i].unsqueeze(1)  # (batch_size, 1, num_tags)

            # 计算现在的分数
            next_score = broadcast_score + self.transitions + broadcast_emission

            # 找出最大分数及其对应的前一个标签
            next_score, indices = next_score.max(dim=1)

            # 根据mask更新
            score = torch.where(masks[:, i].unsqueeze(1).bool(), next_score, score)

            history.append(indices)

        # 添加转移到END的分数
        score = score + self.end_transitions

        # 找出最终最高分数对应的标签
        _, best_tags_list = score.max(dim=1)

        # 回溯得到完整的最优路径
        best_tags = torch.zeros((batch_size, seq_len), dtype=torch.long).to(emissions.device)
        best_tags[:, -1] = best_tags_list

        for i in range(len(history) - 1, -1, -1):
            best_tags[:, i] = history[i].gather(1, best_tags[:, i + 1].unsqueeze(1)).squeeze(1)

        return best_tags


# Cross-Stream Attention
class CrossStreamAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(CrossStreamAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, stream1, stream2, attention_mask=None):
        """Cross-Stream Attention

        Args:
            stream1: (batch_size, seq_len, hidden_size)
            stream2: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = stream1.size()

        # 将stream1作为query，stream2作为key和value
        q = self.query(stream1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(stream2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(stream2).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算attention分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # 应用mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attn_weights, v)

        # 重塑张量
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.out(context)

        return output


# Dual-Stream Networks
class DualStreamNetworks(nn.Module):
    def __init__(self, encoder, hidden_size):
        super(DualStreamNetworks, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size

        # 两个独立的流
        self.stream1_transform = nn.Linear(encoder.config.hidden_size, hidden_size)
        self.stream2_transform = nn.Linear(encoder.config.hidden_size, hidden_size)

        # Cross-Stream Attention
        self.cross_attn_1to2 = CrossStreamAttention(hidden_size)
        self.cross_attn_2to1 = CrossStreamAttention(hidden_size)

        # 融合层
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取基础编码器输出
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = outputs[0]

        # 分成两个流
        stream1 = self.stream1_transform(last_hidden_state)
        stream2 = self.stream2_transform(last_hidden_state)

        # 应用Cross-Stream Attention
        stream1_attended = self.cross_attn_1to2(stream1, stream2, attention_mask)
        stream2_attended = self.cross_attn_2to1(stream2, stream1, attention_mask)

        # 融合两个流
        fused_representation = torch.cat([stream1_attended, stream2_attended], dim=-1)
        fused_representation = self.fusion_layer(fused_representation)

        return fused_representation


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, hidden_size=768, num_edge_types=3, num_tags=None, RoPE=True,
                 use_dual_stream=True, use_ggnn=True, use_crf=True):
        # encoder: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.use_dual_stream = use_dual_stream
        self.use_ggnn = use_ggnn
        self.use_crf = use_crf

        # 添加Dual-Stream Networks
        if use_dual_stream:
            self.dual_stream = DualStreamNetworks(encoder, hidden_size)
            # 适配融合后的表示
            self.fusion_adapter = nn.Linear(hidden_size, self.hidden_size)

        # 添加GGNN
        if use_ggnn:
            self.ggnn = GGNN(hidden_size, num_edge_types)
            # 边的生成器（根据token的表示生成边的关系）
            self.edge_generator = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_edge_types)
            )

        # 添加CRF
        if use_crf and num_tags:
            self.crf = CRF(num_tags)
            self.emission_layer = nn.Linear(self.hidden_size, num_tags)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def create_edge_matrix(self, hidden_states):
        """创建边矩阵用于GGNN

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            edge_matrix: (batch_size, num_edge_types, seq_len, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 为每一对token生成边特征
        edge_features = []
        for i in range(seq_len):
            token_i = hidden_states[:, i:i + 1, :].expand(-1, seq_len, -1)
            token_j = hidden_states
            token_pair = torch.cat([token_i, token_j], dim=-1)
            edge_features.append(token_pair)

        edge_features = torch.stack(edge_features, dim=1)  # (batch_size, seq_len, seq_len, hidden_size*2)
        edge_features = edge_features.view(batch_size, seq_len * seq_len, -1)

        # 应用边生成器
        edge_logits = self.edge_generator(edge_features)  # (batch_size, seq_len*seq_len, num_edge_types)
        edge_logits = edge_logits.view(batch_size, seq_len, seq_len, -1)

        # 通过softmax得到边类型的概率分布
        edge_probs = F.softmax(edge_logits, dim=-1)

        # 转换为所需的形状: (batch_size, num_edge_types, seq_len, seq_len)
        edge_matrix = edge_probs.permute(0, 3, 1, 2)

        return edge_matrix

    def forward(self, input_ids, attention_mask, token_type_ids, tags=None):
        self.device = input_ids.device

        # 获取基础编码器输出
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # 应用Dual-Stream Networks
        if self.use_dual_stream:
            dual_stream_output = self.dual_stream(input_ids, attention_mask, token_type_ids)
            last_hidden_state = self.fusion_adapter(dual_stream_output)

        # 应用GGNN
        if self.use_ggnn:
            # 创建边矩阵
            edge_matrix = self.create_edge_matrix(last_hidden_state)
            # 应用GGNN
            last_hidden_state = self.ggnn(last_hidden_state, edge_matrix)

        # 应用CRF
        crf_output = None
        if self.use_crf:
            emissions = self.emission_layer(last_hidden_state)
            crf_output = self.crf(emissions, attention_mask, tags)

        # GlobalPointer的原始逻辑
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        # 返回结果
        if self.use_crf and tags is not None:
            return logits / self.inner_dim ** 0.5, crf_output
        else:
            return logits / self.inner_dim ** 0.5