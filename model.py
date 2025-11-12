# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math

try:
    # 這是 PyG 的稀疏矩陣乘法
    from torch_sparse import spmm
except ImportError:
    print("警告：未找到 torch_sparse。")
    print("請確保已安裝 PyTorch Geometric: `conda install pyg -c pyg -c pytorch`")
    # 定義一個假的 spmm 以便程式能載入
    def spmm(*args, **kwargs):
        raise ImportError("torch_sparse.spmm 未載入，請安裝 PyG。")


def average_pooling(embeddings: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 4:
        max_len = embeddings.size(2)
        mask = torch.arange(max_len, device=embeddings.device)[None, None, :] < seq_lens.unsqueeze(-1)
        mask = mask.float().unsqueeze(-1)
        masked_sum = torch.sum(embeddings * mask, dim=2)
        count = mask.sum(dim=2) + 1e-9
        return masked_sum / count
    elif embeddings.dim() == 3:
        max_len = embeddings.size(1)
        mask = torch.arange(max_len, device=embeddings.device)[None, :] < seq_lens[:, None]
        mask = mask.float().unsqueeze(-1)
        masked_sum = torch.sum(embeddings * mask, dim=1)
        count = mask.sum(dim=1) + 1e-9
        return masked_sum / count
    else:
        raise ValueError(f"Unsupported embedding dimension: {embeddings.dim()}")

class Hybrid_GNN_MLP(nn.Module):
    """
    (*** V4: 精準對齊 EmbMLP ***)
    此模型 1:1 複製 EmbMLP 的特徵流，
    但將「靜態 ID 查詢」替換為「GNN 快取查詢」。
    """
    def __init__(self, num_users, num_items, hyperparams, 
                 adj_matrix, cates, cate_lens,
                 item_init_vectors=None, cate_init_vectors=None): # (API 保持一致)
        super().__init__()
        self.temperature = hyperparams.get('temperature', 0.07)

        self.num_users = num_users
        self.num_items = num_items
        self.hparams = hyperparams
        self.register_buffer('adj_matrix', adj_matrix)
        
        # --- 1. GNN 的「種子」 (E₀) ---
        # (MODIFIED) 這是「歷史」和「GNN 傳播」的來源
        self.embed_dim = self.hparams.get('embed_dim', 64)
        self.user_embed_dim = self.hparams.get('user_embed_dim', 64)
        self.item_embed_dim = self.hparams.get('item_embed_dim', 64)
        
        # (E₀)
        self.embeddings = nn.Embedding(num_users + num_items, self.embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
        self.num_layers = self.hparams.get('num_layers', 3)

        # --- 2. MLP 的「類別」 (Cate) 特徵 ---
        self.cate_embed_dim = self.hparams.get('cate_embed_dim', 64)
        self.cate_emb_w = nn.Embedding(self.hparams['num_cates'], self.cate_embed_dim)
        nn.init.xavier_uniform_(self.cate_emb_w.weight)
        self.register_buffer('cates', torch.from_numpy(cates))
        self.register_buffer('cate_lens', torch.tensor(cate_lens, dtype=torch.int32))

        # --- 3. 快取 (Caches) ---
        # (a) GNN (e3) 快取
        self.gnn_buffer = nn.Parameter(
            torch.empty(num_users + num_items, self.embed_dim),
            requires_grad=False 
        )
        
        # (b) History 快取 (*** 與 EmbMLP 完全對齊 ***)
        # (hist_item_emb + hist_cate_emb)
        history_embed_dim = self.item_embed_dim + self.cate_embed_dim
        self.user_history_buffer = nn.Embedding(num_users, history_embed_dim)
        self.user_history_buffer.weight.requires_grad = False

        # --- 4. 建立「MLP 塔」 (*** 與 EmbMLP 完全對齊 ***)
        user_concat_dim = self.user_embed_dim + self.item_embed_dim + self.cate_embed_dim
        item_concat_dim = self.item_embed_dim + self.cate_embed_dim + self.user_embed_dim
        
        layer_num = 1
        self.user_mlp = nn.Sequential(
                *[self.PreNormResidual(user_concat_dim, self._build_mlp_layers(user_concat_dim))]*layer_num,
            )
        self.item_mlp = nn.Sequential(
                *[self.PreNormResidual(item_concat_dim, self._build_mlp_layers(item_concat_dim))]*layer_num,
            )
        self.user_mlp_2 = nn.Sequential(
                *[self.PreNormResidual(user_concat_dim, self._build_mlp_layers(user_concat_dim))]*layer_num,
            )
        self.item_mlp_2 = nn.Sequential(
                *[self.PreNormResidual(item_concat_dim, self._build_mlp_layers(item_concat_dim))]*layer_num,
            )

    # (PreNormResidual 和 _build_mlp_layers 和 EmbMLP 一樣)
    class PreNormResidual(nn.Module):
        def __init__(self, dim, fn): super().__init__(); self.fn = fn; self.norm = nn.LayerNorm(dim)
        def forward(self, x): return self.fn(self.norm(x)) + x
    def _build_mlp_layers(self, dim, expansion_factor = 2, dropout = 0., dense = nn.Linear):
        inner_dim = int(dim * expansion_factor); return nn.Sequential(dense(dim, inner_dim), nn.ReLU(), nn.Dropout(dropout), dense(inner_dim, dim), nn.Dropout(dropout))
        
    def _run_gnn_propagation(self) -> torch.Tensor:
        # (GNN 傳播邏輯不變)
        e_0 = self.embeddings.weight
        embeddings_k = [e_0]
        for layer in range(self.num_layers):
            e_k = torch.sparse.mm(self.adj_matrix, embeddings_k[-1])
            embeddings_k.append(e_k)
        final_embeddings = torch.mean(torch.stack(embeddings_k, dim=0), dim=0)
        return final_embeddings

    def update_gnn_buffer(self):
        # (「慢心跳」快取更新，邏輯不變)
        with torch.no_grad():
            e_final = self._run_gnn_propagation()
            self.gnn_buffer.data.copy_(e_final)
            
    def _get_user_features(self, batch: Dict) -> torch.Tensor:
        """
        (*** MODIFIED: 與 EmbMLP 對齊 ***)
        """
        # 1. 靜態 User 特徵 (*** 核心替換 ***)
        # (B, D_u)
        # static_u_emb = self.gnn_buffer[batch['users']]
        static_u_emb = self._run_gnn_propagation()[batch['users']]

        # 2. 動態 History 特徵
        # (B, T, D_i)
        hist_item_emb = self.embeddings(batch['item_history_matrix'] + self.num_users)
        # (B, T, C_max, D_c)
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        # (B, T, D_c)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len)
        # (B, T, D_i + D_c)
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)
        # (B, D_i + D_c)
        user_history_emb = average_pooling(hist_item_emb_with_cate, batch['item_history_len'])

        # 3. (NEW) 更新 History 快取 (為了 inference)
        if self.training:
            self.user_history_buffer.weight[batch['users']] = user_history_emb.detach()

        # 4. 融合 (Concat)
        # (B, D_u + D_i + D_c)
        user_features = torch.cat([static_u_emb, user_history_emb], dim=-1)
        return user_features

    def _get_item_features(self, batch: Dict) -> torch.Tensor:
        """
        (*** MODIFIED: 與 EmbMLP 對齊 ***)
        """
        # 1. 靜態 Item 特徵 (*** 核心替換 ***)
        # (B, D_i)
        # static_item_emb = self.gnn_buffer[batch['items'] + self.num_users]
        static_item_emb = self._run_gnn_propagation()[batch['items'] + self.num_users]

        # 2. 靜態 Cate 特徵
        # (B, C_max, D_c)
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        # (B, D_c)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        # (B, D_i + D_c)
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)

        # 3. 動態 History 特徵 (Item 被哪些 User 看過)
        # (B, T_u, D_u)
        item_history_user_emb = self.embeddings(batch['user_history_matrix'])
        # (B, D_u)
        item_history_emb = average_pooling(item_history_user_emb, batch['user_history_len'])

        # 4. 融合 (Concat)
        # (B, D_i + D_c + D_u)
        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)
        return item_features

    def _get_embeddings_from_features(self, user_features, item_features):
        """ (*** 與 EmbMLP 對齊 ***) """
        user_embedding = self.user_mlp(user_features)
        user_embedding_2 = self.user_mlp_2(user_features)
        user_embedding =  user_embedding + user_embedding_2
        item_embedding = self.item_mlp(item_features)
        item_embedding_2 = self.item_mlp_2(item_features)
        item_embedding =  item_embedding + item_embedding_2
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        return user_embedding, item_embedding

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """ (*** 與 EmbMLP 對齊 ***) """
        user_features = self._get_user_features(batch)
        item_features = self._get_item_features(batch)
        user_embedding, item_embedding = self._get_embeddings_from_features(user_features, item_features)
        return user_embedding, item_embedding
    
    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
        """計算 InfoNCE 損失 (與 TF1 數學邏輯完全對齊的最終修正版)"""
        
        # --- 分母 (Denominator) 計算 ---
        # 1. 計算 item-user 分數矩陣 [B, B]，M[i,j] = item_i @ user_j
        #    這與 TF1 的 tf.matmul(item, user, transpose_b=True) 完全等價
        all_inner_product = torch.matmul(item_embedding, user_embedding.t())
        logits = all_inner_product / self.temperature
        
        # 2. 應用 Log-Sum-Exp 技巧穩定化
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stabilized = logits - logits_max
        exp_logits_den = torch.exp(logits_stabilized)
        
        # 3. 分母是穩定後 exp_logits 的行和 (對每個 item，匯總所有 user 的分數)
        denominator = exp_logits_den.sum(dim=1, keepdim=True)

        # --- 分子 (Numerator) 計算 ---
        # 1. 獨立計算正樣本對 (user_i, item_i) 的分數
        pred_scores = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        pred_logits = pred_scores / self.temperature
        
        # 2. 使用與分母相同的 logits_max 來穩定化分子
        #    確保分子和分母的縮放標準一致
        pred_logits_stabilized = pred_logits - logits_max
        numerator = torch.exp(pred_logits_stabilized)

        # --- 最終計算 ---
        infonce_pred = (numerator / (denominator + 1e-9)).squeeze()
        infonce_pred = torch.clamp(infonce_pred, min=1e-9, max=1.0 - 1e-9)
        
        return F.binary_cross_entropy(infonce_pred, labels.float(), reduction='none')

    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        # (InfoNCE + L2 Reg 邏輯不變)
        user_embedding, item_embedding = self.forward(batch)
        losses = self._calculate_infonce_loss(user_embedding, item_embedding, batch['labels'])
        loss = losses.mean()
        l2_reg = self.hparams.get('l2_reg', 1e-4)
        if l2_reg > 0 and self.training:
            # (MODIFIED) 梯度只來自 E₀ 和 E_cate
            reg_loss = self.embeddings(batch['users']).norm(2).pow(2) + \
                       self.embeddings(batch['items'] + self.num_users).norm(2).pow(2) + \
                       self.cate_emb_w(self.cates[batch['items']]).norm(2).pow(2)
            loss += l2_reg * (reg_loss / len(batch['users']))
        return loss
    
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_user_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (*** MODIFIED: 與 EmbMLP 對齊的快取推論 ***)
        """
        # --- 1. 計算「正樣本」 (B, D) ---
        user_features, item_features = self._get_user_features(batch), self._get_item_features(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        
        # --- 2. 計算「負樣本」特徵 (B, M, D_mlp_in) ---
        num_neg_samples = neg_user_ids_batch.shape[1]
        
        # (a) 靜態 GNN 特徵 (查快取)
        neg_user_gnn = self.gnn_buffer[neg_user_ids_batch] # (B, M, D_u)
        
        # (b) 動態 History 特徵 (查快取 !!)
        neg_user_hist = self.user_history_buffer(neg_user_ids_batch) # (B, M, D_i + D_c)

        # --- 3. 融合 (Concat) ---
        neg_user_features = torch.cat([neg_user_gnn, neg_user_hist], dim=-1) # (B, M, D_u + D_i + D_c)
        
        # --- 4. 過 MLP ---
        # (我們需要 item_features_expanded)
        item_features_expanded = item_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        
        neg_user_emb, neg_item_emb = self._get_embeddings_from_features(
            neg_user_features, 
            item_features_expanded
        )
        
        # --- 5. 計算分數 ---
        neg_logits = torch.sum(neg_user_emb * neg_item_emb, dim=2)
        
        return pos_logits, neg_logits, torch.tensor(0.0)
    
# class LightGCN(nn.Module):
#     """
#     LightGCN 模型的 PyTorch 實現。
#     """
#     def __init__(self, num_users: int, num_items: int, hyperparams: Dict,
#                  adj_matrix: torch.sparse.FloatTensor):
#         """
#         Args:
#             num_users (int): 用戶總數
#             num_items (int): 物品總數
#             hyperparams (Dict): 包含 'embed_dim' 和 'num_layers' 的字典
#             adj_matrix (torch.sparse.FloatTensor): 
#                 (N_u + N_i) x (N_u + N_i) 的「對稱正規化」鄰接矩陣
#         """
#         super().__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embed_dim = hyperparams.get('embed_dim', 64)
#         self.num_layers = hyperparams.get('num_layers', 3)
#         self.hparams = hyperparams
        
#         print(f"--- Initializing LightGCN ---")
#         print(f"  Users: {self.num_users}, Items: {self.num_items}")
#         print(f"  Embed Dim: {self.embed_dim}, Layers: {self.num_layers}")
        
#         # 1. 唯一的「可訓練」參數：E_0
#         self.embeddings = nn.Embedding(
#             num_users + num_items, 
#             self.embed_dim
#         )
#         # (我們也把 user_history_buffer 留著，以防萬一，雖然 GNN 不太用它)
#         self.user_history_buffer = nn.Embedding(num_users, self.embed_dim)

#         # 2. 註冊「圖」 (它不是參數，是 buffer)
#         self.register_buffer('adj_matrix', adj_matrix)

#         self._init_weights()

#     def _init_weights(self):
#         # LightGCN 使用 Xavier 初始化 E_0
#         nn.init.xavier_uniform_(self.embeddings.weight)
#         print("--- LightGCN Embeddings initialized (Xavier) ---")

#     def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         執行 LightGCN 的 K 層圖卷積 (Graph Convolution)。
        
#         這是在「全圖」上操作的，它*沒有* batch 參數。
#         它在訓練迴圈中每個 epoch 只被呼叫一次。
        
#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: 
#                 (users_final_emb, items_final_emb)
#         """
        
#         # E_k: 儲存每一層的 Embedding
#         all_embeddings = self.embeddings.weight
#         embeddings_k = [all_embeddings] # 第 0 層 (E_0)
        
#         # --- GNN 傳播 (K 層) ---
#         for layer in range(self.num_layers):
#             # 核心邏輯: E_{k+1} = (D^{-1/2} A D^{-1/2}) @ E_k
#             # all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
#             all_embeddings = spmm(
#                 self.adj_matrix.indices(), 
#                 self.adj_matrix.values(), 
#                 self.adj_matrix.shape[0], 
#                 self.adj_matrix.shape[1], 
#                 all_embeddings
#             )
#             embeddings_k.append(all_embeddings)

#         # --- 最終輸出 (LightGCN 的秘方) ---
#         # 最終的 Embedding E_final 是 K+1 層的「加權平均」
#         # (標準 LightGCN 是「平均」，我們這裡用 1/(K+1))
#         final_embeddings = torch.mean(torch.stack(embeddings_k, dim=0), dim=0)
        
#         # 切分回 user 和 item
#         users_emb, items_emb = torch.split(
#             final_embeddings, 
#             [self.num_users, self.num_items]
#         )
        
#         # 在訓練時，順便更新一下 history buffer (為了舊的評估邏輯)
#         if self.training:
#             self.user_history_buffer.weight.data = users_emb.detach()
        
#         return users_emb, items_emb

#     def calculate_loss(self, users_emb: torch.Tensor, items_emb: torch.Tensor, 
#                        batch: Dict) -> torch.Tensor:
#         """
#         計算 LightGCN 的 BPR Loss (Bayesian Personalized Ranking)。
        
#         Args:
#             users_emb (torch.Tensor): 來自 forward() 的「最終」用戶向量
#             items_emb (torch.Tensor): 來自 forward() 的「最終」物品向量
#             batch (Dict): 來自 BPRDataset 的批次，包含 (u, i, j)
#         """
        
#         # 1. 獲取 BPR 批次
#         user_ids = batch['user_id']
#         pos_item_ids = batch['pos_item_id']
#         neg_item_ids = batch['neg_item_id']

#         # 2. 查找 GNN 產生的最終 embedding
#         u_emb = users_emb[user_ids]
#         pos_i_emb = items_emb[pos_item_ids]
#         neg_i_emb = items_emb[neg_item_ids]

#         # 3. BPR Loss: -log(sigmoid(pos_score - neg_score))
#         pos_score = torch.sum(u_emb * pos_i_emb, dim=1)
#         neg_score = torch.sum(u_emb * neg_i_emb, dim=1)
        
#         # F.softplus(x) = log(1 + exp(x))
#         # BPR Loss = mean( -log( sigmoid(pos - neg) ) )
#         #          = mean( -log( 1 / (1 + exp(neg - pos)) ) )
#         #          = mean( log( 1 + exp(neg - pos) ) )
#         #          = mean( softplus( neg - pos ) )
#         loss = F.softplus(neg_score - pos_score).mean()
        
#         # (可選) 加上 L2 正規化 (Weight Decay)
#         l2_reg = self.hparams.get('l2_reg', 1e-4)
#         if l2_reg > 0:
#             # 只對「當前批次」用到的 E_0 向量進行正規化
#             # (這是 LightGCN 論文的推薦作法)
#             u_emb_0 = self.embeddings(user_ids)
#             pos_i_emb_0 = self.embeddings(pos_item_ids + self.num_users)
#             neg_i_emb_0 = self.embeddings(neg_item_ids + self.num_users)
            
#             reg_loss = (u_emb_0.norm(2).pow(2) + 
#                         pos_i_emb_0.norm(2).pow(2) + 
#                         neg_i_emb_0.norm(2).pow(2)) / float(len(user_ids))
            
#             loss = loss + l2_reg * reg_loss

#         return loss

    
#     def inference_for_evaluation(self, 
#                                  users_emb: torch.Tensor, 
#                                  items_emb: torch.Tensor,
#                                  batch: Dict, 
#                                  neg_user_ids_batch: torch.Tensor
#                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         (*** 新函式 ***)
#         為了和你「舊的」評估標準 (99 neg_users vs 1 pos_item) 保持一致。
#         """
        
#         # a. 獲取「正樣本」向量
#         # (B, D)
#         pos_user_emb = users_emb[batch['users']]
#         # (B, D)
#         pos_item_emb = items_emb[batch['items']]

#         # b. 獲取「負樣本」向量
#         # (B, M, D)
#         neg_user_emb = users_emb[neg_user_ids_batch]
        
#         # c. 計算分數
#         # 正樣本分數 (u, i)
#         pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1) # (B,)
        
#         # 負樣本分數 (neg_u, i)
#         # 擴展 item 向量: (B, D) -> (B, 1, D)
#         neg_logits = torch.sum(neg_user_emb * pos_item_emb.unsqueeze(1), dim=2) # (B, M)
        
#         return pos_logits, neg_logits

