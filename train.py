# -*- coding: utf-8 -*-
from data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from TModel import GlobalPointer, MetricsCalculator
from tqdm import tqdm

bert_model_path = 'MacBERT'
train_cme_path = './datasets/CHIP-STS/CHIP-STS_train.json'  # 训练集
eval_cme_path = './datasets/CHIP-STS/CHIP-STS_dev.json'  # 验证集
device = torch.device("cuda:0")
BATCH_SIZE = 16

ENT_CLS_NUM = 9
NUM_TAGS = 2  # 二分类: 是实体/不是实体
HIDDEN_SIZE = 768  # 与MacBERT模型的hidden_size一致
NUM_EDGE_TYPES = 3  # 边的类型数量，根据任务调整

# tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

# train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True,
                              num_workers=16)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)

# GP MODEL
encoder = BertModel.from_pretrained(bert_model_path)
# 初始化模型
model = GlobalPointer(
    encoder=encoder,
    ent_type_size=ENT_CLS_NUM,
    inner_dim=64,
    hidden_size=HIDDEN_SIZE,
    num_edge_types=NUM_EDGE_TYPES,
    num_tags=NUM_TAGS,
    RoPE=True,
    use_dual_stream=True,
    use_ggnn=True,
    use_crf=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred, crf_loss=None):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    crf_loss: CRF损失值（如果使用CRF）
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    gp_loss = multilabel_categorical_crossentropy(y_true, y_pred)

    # 有CRF损失，加入总损失
    if crf_loss is not None:
        # 调整CRF损失的权重
        total_loss = gp_loss + 0.5 * crf_loss
        return total_loss
    return gp_loss


# 生成序列标签
def generate_sequence_labels(labels):
    """从原始标签生成序列标签
    labels: (batch_size, ent_type_size, seq_len, seq_len)
    返回: (batch_size, seq_len)，每个位置为0或1，1表示该位置是实体的一部分
    """
    batch_size, ent_type_size, seq_len, _ = labels.shape
    sequence_labels = torch.zeros((batch_size, seq_len), dtype=torch.long).to(labels.device)

    # 如果任何实体类型在该位置标记为实体的一部分，则设置为1
    for b in range(batch_size):
        for i in range(seq_len):
            # 检查是否有任何实体类型在该位置有标记
            if torch.any(labels[b, :, i, :] > 0) or torch.any(labels[b, :, :, i] > 0):
                sequence_labels[b, i] = 1

    return sequence_labels


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)

        # 生成序列标签
        sequence_labels = generate_sequence_labels(labels)

        # 模型前向传播
        if model.use_crf:
            logits, crf_loss = model(input_ids, attention_mask, segment_ids, sequence_labels)
            loss = loss_fun(labels, logits, crf_loss)
        else:
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(labels, logits)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        if (idx + 1) % 10 == 0:  # 每10个batch打印一次
            print(
                f"Epoch {eo + 1}/{10}, Batch {idx + 1}/{len(ner_loader_train)}, Train Loss: {avg_loss:.4f}, Train F1: {avg_f1:.4f}")

    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Validating"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)

            # 生成序列标签（用于CRF）
            sequence_labels = generate_sequence_labels(labels)

            # 模型前向传播
            if model.use_crf:
                logits, _ = model(input_ids, attention_mask, segment_ids, sequence_labels)
            else:
                logits = model(input_ids, attention_mask, segment_ids)

            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r

        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        print("EPOCH：{}\tEVAL_F1:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\t".format(eo + 1, avg_f1, avg_precision,
                                                                                   avg_recall))

        if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/ent_model.pth')
            max_f = avg_f1
            print(f"Model saved with new best F1: {max_f:.4f}")

        model.train()

print(f"Training completed. Best F1: {max_f:.4f}")