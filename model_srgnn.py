# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math
from model_mlp import EmbMLP

# ===================================================================
# 輔助函式 
# ===================================================================
    
def average_pooling(embeddings: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    對 Embedding 進行帶遮罩的平均池化 (PyTorch 版本)。
    """
    if embeddings.dim() == 4: # e.g., [B, T, C_max, E]
        # seq_lens: [B, T] -> mask: [B, T, C_max]
        max_len = embeddings.size(2)
        mask = torch.arange(max_len, device=embeddings.device)[None, None, :] < seq_lens.unsqueeze(-1)
        mask = mask.float().unsqueeze(-1) # [B, T, C_max, 1]
        
        masked_sum = torch.sum(embeddings * mask, dim=2)
        count = mask.sum(dim=2) + 1e-9
        return masked_sum / count
    
    elif embeddings.dim() == 3: # e.g., [B, T, E]
        # seq_lens: [B] -> mask: [B, T]
        max_len = embeddings.size(1)
        mask = torch.arange(max_len, device=embeddings.device)[None, :] < seq_lens[:, None]
        mask = mask.float().unsqueeze(-1) # [B, T, 1]
        
        masked_sum = torch.sum(embeddings * mask, dim=1)
        count = mask.sum(dim=1) + 1e-9
        return masked_sum / count
    
    else:
        raise ValueError(f"Unsupported embedding dimension: {embeddings.dim()}")

# ===================================================================
# (*** NEW ***) SR-GNN (Session-based Rec GNN)
# ===================================================================
class SR_GNN_Model(nn.Module):
    """
    SR-GNN 論文的 "純粹" 實作。
    
    - 結構: Session-based (單塔)，只使用 itemSeq。
    - 忽略: userId, userSeq, cateId (論文不使用類別)。
    - GNN: Gated GNN (GGNN)，n_layers=1。
    - 聚合: Gated Attention (h_last + h_global)。
    - 預測: dot(h_session, item_embedding)。
    - Loss: 繼承 main_model_utils.py 的 InfoNCE Loss。
    """
    
    def __init__(self, 
                 cates: np.ndarray, # (未使用，但為了 API 一致性保留)
                 cate_lens: np.ndarray, # (未使用)
                 hyperparams: Dict, 
                 train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        super().__init__()
        self.hparams = hyperparams
        
        # 1. 獲取核心參數
        self.num_items = self.hparams['num_items']
        self.embed_dim = self.hparams['item_embed_dim'] # 假設 item_embed_dim
        
        # 2. 建立靜態 Item Embedding (這是模型唯一需要的 Embedding)
        self.item_emb_w = nn.Embedding(self.num_items, self.embed_dim)

        # 3. 建立 GGNN (Gated GNN) 層 (論文 n_layers=1)
        # 論文 Eq(4): a = A * Emb * W_H
        # 我們的 A 分為 A_in, A_out。
        # 實現: m = Concat(A_in @ H, A_out @ H) @ W_H
        self.W_H = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True)
        self.gnn_gru = nn.GRUCell(self.embed_dim, self.embed_dim)

        # 4. 建立 Gated Attention 聚合層 (論文 Eq(6, 7, 8))
        self.W_1 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.W_2 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_vec = nn.Parameter(torch.Tensor(self.embed_dim)) # q in Eq(6)
        self.b_attn = nn.Parameter(torch.Tensor(self.embed_dim)) # c in Eq(6)
        
        # 5. 建立最終 Session Embedding 投影 (論文 Eq(9))
        self.W_3 = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True)
        
        # 6. Loss 參數 (為了 InfoNCE)
        self.temperature = hyperparams.get('temperature', 0.07)

        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb_w.weight)
        nn.init.xavier_uniform_(self.W_H.weight)
        nn.init.zeros_(self.W_H.bias)
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.zeros_(self.W_1.bias)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.zeros_(self.W_2.bias)
        nn.init.xavier_uniform_(self.W_3.weight)
        nn.init.zeros_(self.W_3.bias)
        nn.init.zeros_(self.q_vec)
        nn.init.zeros_(self.b_attn)
        # GRUCell 權重 PyTorch 會自動初始化

    def _build_batched_adj(self, seq_lens: torch.Tensor, max_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """ (從 SR_GNN_Model 複製而來，邏輯相同) """
        B = seq_lens.size(0)
        A_in = torch.zeros(B, max_len, max_len, device=device, dtype=torch.float32)
        A_out = torch.zeros(B, max_len, max_len, device=device, dtype=torch.float32)
        
        for i in range(B):
            l = seq_lens[i].item()
            if l <= 1:
                continue
            idx_in_row = torch.arange(1, l, device=device)
            idx_in_col = torch.arange(0, l-1, device=device)
            A_in[i, idx_in_row, idx_in_col] = 1.0
            idx_out_row = torch.arange(0, l-1, device=device)
            idx_out_col = torch.arange(1, l, device=device)
            A_out[i, idx_out_row, idx_out_col] = 1.0
        return A_in, A_out

    def _run_ggnn(self, seq_emb: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """ 
        執行 GGNN (n_layers=1)，還原論文 Eq(4, 5) 
        seq_emb: [B, T, D]
        """
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        A_in, A_out = self._build_batched_adj(seq_lens, T, device)
        
        h = seq_emb # (B, T, D)
        
        # 論文 Eq(4)
        m_in = torch.bmm(A_in, h)  # [B, T, T] @ [B, T, D] -> [B, T, D]
        m_out = torch.bmm(A_out, h)
        
        # (B, T, D*2)
        m_concat = torch.cat([m_in, m_out], dim=2)
        
        # (B, T, D*2) @ (D*2, D) -> (B, T, D)
        m_agg = self.W_H(m_concat) # m_agg = a in Eq(4)
        
        # 論文 Eq(5): GRUCell
        # (B, T, D) -> (B*T, D)
        h_flat = h.view(-1, D)
        m_agg_flat = m_agg.view(-1, D)
        
        h_new_flat = self.gnn_gru(m_agg_flat, h_flat)
        
        # (B, T, D)
        h_new = h_new_flat.view(B, T, D)
        
        # Masking: 只保留非 padding 部分的更新
        mask = (torch.arange(T, device=device)[None, :] < seq_lens[:, None]).unsqueeze(-1)
        h_final = torch.where(mask, h_new, h) # h_final = v in Eq(5)
        
        return h_final

    def _run_sr_aggregation(self, gnn_out: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """ 
        執行 Gated Attention 聚合，還原論文 Eq(6, 7, 8, 9) 
        gnn_out: [B, T, D] (GNN 處理後的序列)
        """
        B, T, D = gnn_out.shape
        device = gnn_out.device

        # 1. 獲取 s_local (h_last)，論文 Eq(7)
        # (B,) -> (B, 1, 1) -> (B, 1, D)
        last_indices = (seq_lens - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, D)
        # (B, D)
        h_last = torch.gather(gnn_out, 1, last_indices).squeeze(1) 
        
        # 2. 計算 Gated Attention (論文 Eq(6))
        # W_1 * h_i
        wh_i = self.W_1(gnn_out) # (B, T, D)
        # W_2 * h_last
        wh_last = self.W_2(h_last).unsqueeze(1) # (B, 1, D)
        
        # sigmoid( ... + c)
        gate_hidden = torch.sigmoid(wh_i + wh_last + self.b_attn) # (B, T, D)
        
        # alpha_i = q^T * gate_hidden
        attn_scores = (gate_hidden * self.q_vec).sum(dim=-1) # (B, T)
        
        # 3. 應用 Padding Mask
        mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]
        attn_scores.masked_fill_(~mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1) # (B, T, 1)
        
        # 4. 獲取 s_global (論文 Eq(8))
        # (B, T, 1) * (B, T, D) -> sum(dim=1) -> (B, D)
        h_global = (attn_weights * gnn_out).sum(dim=1)
        
        # 5. 獲取 s_hybrid (論文 Eq(9))
        h_session = self.W_3(torch.cat([h_last, h_global], dim=1))
        
        return h_session # (B, D)

    def _get_session_embedding(self, batch: Dict) -> torch.Tensor:
        """
        輔助函式：執行完整的 SR-GNN 流程
        """
        # 1. 獲取靜態 Item Embedding
        # (B, T, D)
        seq_emb = self.item_emb_w(batch['item_history_matrix'])
        
        # 2. 執行 GNN
        # (B, T, D)
        ggnn_out = self._run_ggnn(seq_emb, batch['item_history_len'])
        
        # 3. 執行聚合
        # (B, D)
        h_session = self._run_sr_aggregation(ggnn_out, batch['item_history_len'])
        
        return h_session

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播：計算 Session Embedding 和 Target Item Embedding
        """
        # 1. User Tower (Session Tower)
        # h_session: (B, D)
        h_session = self._get_session_embedding(batch)
        
        # 2. Item Tower (Target Item)
        # item_emb: (B, D)
        item_emb = self.item_emb_w(batch['items'])
        
        # 3. L2 歸一化 (為了 InfoNCE/Cosine Similarity)
        h_session = F.normalize(h_session, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        
        return h_session, item_emb

    # (*** 複製 EmbMLP 的 InfoNCE Loss ***)
    # 我們使用您框架中的 InfoNCE Loss (BCE 形式)
    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
        """ (*** 複製自 EmbMLP ***) """
        all_inner_product = torch.matmul(user_embedding, item_embedding.t())
        logits = all_inner_product / self.temperature
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stabilized = logits - logits_max
        exp_logits_den = torch.exp(logits_stabilized)
        denominator = exp_logits_den.sum(dim=1, keepdim=True)
        
        pred_scores = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        pred_logits = pred_scores / self.temperature
        pred_logits_stabilized = pred_logits - logits_max
        numerator = torch.exp(pred_logits_stabilized)

        infonce_pred = (numerator / (denominator + 1e-9)).squeeze()
        infonce_pred = torch.clamp(infonce_pred, min=1e-9, max=1.0 - 1e-9)
        
        return F.binary_cross_entropy(infonce_pred, labels.float(), reduction='none')
    
    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        """
        (*** MODIFIED: 使用 InfoNCE Loss ***)
        """
        # (B, D), (B, D)
        session_embedding, item_embedding = self.forward(batch)
        
        losses = self._calculate_infonce_loss(session_embedding, item_embedding, batch['labels'])
        final_loss = losses.mean()
        
        return final_loss
        
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (*** 核心修改 ***)
        """
        # --- 1. 計算 Session Embedding ---
        # (B, D)
        h_session = self._get_session_embedding(batch)
        h_session_norm = F.normalize(h_session, p=2, dim=-1)

        # --- 2. 獲取正樣本 Embedding ---
        # (B, D)
        pos_i_emb = self.item_emb_w(batch['items'])
        pos_i_emb_norm = F.normalize(pos_i_emb, p=2, dim=-1)
        
        # --- 3. 計算正樣本 Logits (Cosine Similarity) ---
        # (B,)
        pos_logits = torch.sum(h_session_norm * pos_i_emb_norm, dim=1)

        # --- 4. 獲取負樣本 Embedding ---
        # (B, M, D)
        neg_i_emb = self.item_emb_w(neg_item_ids_batch)
        neg_i_emb_norm = F.normalize(neg_i_emb, p=2, dim=-1)
        
        # --- 5. 計算負樣本 Logits ---
        # (B, 1, D)
        h_session_expanded = h_session_norm.unsqueeze(1)
        # (B, M)
        neg_logits = torch.sum(h_session_expanded * neg_i_emb_norm, dim=2)
        
        # --- 6. 計算 Loss (用於指標) ---
        # (使用 "未歸一化" 的點積來計算 BCE Loss，與 EmbMLP 保持一致)
        pos_logits_for_loss = torch.sum(h_session * pos_i_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits_for_loss, batch['labels'].float(), reduction='none'
        )
        
        # --- 7. 返回結果 (返回 L2 歸一化後的 logits，用於排名) ---
        return pos_logits, neg_logits, per_sample_loss