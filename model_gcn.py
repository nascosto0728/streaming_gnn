import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math

from model_mlp import EmbMLP

   
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


class Hybrid_GNN_MLP(EmbMLP):
    """
    Hybrid GNN-MLP 模型。
    它繼承自 EmbMLP，重用了 MLP 層和 Loss 計算，
    但覆寫了特徵建構 (forward) 和推論 (inference) 邏輯，
    以 LightGCN 產生的 Embedding 取代靜態 ID Embedding。
    """
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 hyperparams: Dict, 
                 adj_matrix: torch.Tensor, # 傳入 GNN 用的圖
                 cates_np: np.ndarray, 
                 cate_lens_np: np.ndarray):
        
        # 1. 初始化父類別 (EmbMLP)
        # 我們需要傳入 EmbMLP 所需的參數
        super().__init__(
            cates=cates_np,
            cate_lens=cate_lens_np,
            hyperparams=hyperparams,
            train_config={},  # train_config 在 EmbMLP 的 init 中未被使用，傳入空字典
            item_init_vectors=None, # GNN 將取代 SBERT/KGE
            cate_init_vectors=None
        )
        
        # 2. 儲存 GNN 相關參數
        self.num_users = num_users
        self.num_items = num_items
        self.adj_matrix = adj_matrix # 儲存傳入的稀疏鄰接矩陣
        self.n_layers = self.hparams.get('n_layers', 2) # 從 config 讀取 GNN 層數
        
        print(f"[Hybrid_GNN_MLP] Initialized with {self.n_layers} GNN layers.")

        # 3. 註冊 GNN 緩存 (Buffer)，用於 "評估"
        # 這些 Buffer 將在 update_gnn_buffer() 中被填充
        self.register_buffer(
            'gnn_user_emb_buffer', 
            torch.empty_like(self.user_emb_w.weight, device=self.user_emb_w.weight.device)
        )
        self.register_buffer(
            'gnn_item_emb_buffer', 
            torch.empty_like(self.item_emb_w.weight, device=self.item_emb_w.weight.device)
        )

    def _propagate_lightgcn(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        執行 LightGCN 的圖傳播。
        這個函式會保留梯度，以便在訓練時進行端到端優化。
        """
        # 1. 獲取初始 Embedding (可訓練的)
        user_emb_0 = self.user_emb_w.weight
        item_emb_0 = self.item_emb_w.weight
        
        all_emb = torch.cat([user_emb_0, item_emb_0], dim=0)
        embs_list = [all_emb] # 第 0 層

        # 2. 執行 GNN 傳播
        for _ in range(self.n_layers):
            # 核心：稀疏矩陣乘法
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs_list.append(all_emb)
            
        # 3. LightGCN 的標準做法：對所有層的 Embedding 取平均
        final_embs = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        
        # 4. 拆分為 User 和 Item
        final_user_embs, final_item_embs = torch.split(
            final_embs, [self.num_users, self.num_items]
        )
        
        return final_user_embs, final_item_embs

    def update_gnn_buffer(self):
        """
        (供 main_model_utils.py 呼叫)
        計算 GNN embedding 並將其 "凍結" 到 buffer 中，
        專門用於快速的 "評估" 階段。
        """
        print("--- (Hybrid) Updating GNN buffer for evaluation ---")
        with torch.no_grad():
            user_embs, item_embs = self._propagate_lightgcn()
            # .data.copy_() 是原地操作，確保 buffer 被更新
            self.gnn_user_emb_buffer.data.copy_(user_embs.detach())
            self.gnn_item_emb_buffer.data.copy_(item_embs.detach())

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (*** 覆寫 ***) Hybrid 模型的訓練前向傳播。
        """
        # --- 1. (GNN) 執行圖傳播 (保持梯度) ---
        # 這是與 EmbMLP 最大的不同：Embedding 是動態計算的
        gnn_user_embs, gnn_item_embs = self._propagate_lightgcn()

        # --- 2. (MLP) 建構特徵 (邏輯同 EmbMLP._build_feature_representations) ---
        
        # === 使用者表示 ===
        # (*** MODIFIED ***) 使用 GNN 傳播後的 Embedding
        static_u_emb = gnn_user_embs[batch['users']]
        u_emb = static_u_emb 
        
        # (*** MODIFIED ***) 歷史序列也使用 GNN Embedding
        hist_item_emb = gnn_item_embs[batch['item_history_matrix']]
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates) # 類別 Embedding 保持不變
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len)
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)

        user_history_emb = average_pooling(hist_item_emb_with_cate, batch['item_history_len']).detach()
        user_features = torch.cat([u_emb, user_history_emb], dim=-1)

        # === 物品表示 ===
        # (*** MODIFIED ***) 使用 GNN 傳播後的 Embedding
        static_item_emb = gnn_item_embs[batch['items']]
        item_emb = static_item_emb

        # 類別
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        item_emb_with_cate = torch.cat([item_emb, avg_cate_emb_for_item], dim=1)
        
        # (*** MODIFIED ***) 用戶歷史也使用 GNN Embedding
        item_history_user_emb = gnn_user_embs[batch['user_history_matrix']]
        item_history_emb = average_pooling(item_history_user_emb, batch['user_history_len']).detach()

        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)

        # (*** 繼承 ***)
        # 我們仍然更新 EmbMLP 中的 item_history_buffer
        # 雖然 GNN 模型在推論時不依賴它，但這樣做可以保持與父類的一致性
        if self.training:
            self.user_history_buffer.weight[batch['items']] = item_history_emb.detach()

        # --- 3. (MLP) 執行 MLP 和歸一化 (重用父類別的方法) ---
        user_embedding, item_embedding = self._get_embeddings_from_features(user_features, item_features)
        
        return user_embedding, item_embedding
    
    # def calculate_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    #   """
    #   (*** 無需覆寫 ***)
    #   父類別 EmbMLP 的 calculate_loss 會自動呼叫我們覆寫的 forward()，
    #   並使用相同的 InfoNCE 損失函式，因此我們繼承它即可。
    #   """
    #   return super().calculate_loss(batch)
        
    
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (*** 覆寫 ***) Hybrid 模型的 "評估" 推論邏輯。
        
        主要區別：
        1. 使用 self.gnn_user_emb_buffer 和 self.gnn_item_emb_buffer (預先算好，無梯度)。
        2. 不再依賴 EmbMLP 的 item_history_buffer (GNN buffer 已包含歷史)。
        """
        
        # --- 1. 計算正樣本的 Embedding 和 Logits (使用 GNN Buffers) ---
        
        # === 使用者特徵 (來自 Buffer) ===
        static_u_emb = self.gnn_user_emb_buffer[batch['users']]
        hist_item_emb = self.gnn_item_emb_buffer[batch['item_history_matrix']]
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len)
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)
        user_history_emb = average_pooling(hist_item_emb_with_cate, batch['item_history_len']).detach()
        user_features = torch.cat([static_u_emb, user_history_emb], dim=-1)

        # === 物品特徵 (來自 Buffer) ===
        static_item_emb = self.gnn_item_emb_buffer[batch['items']]
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len)
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        item_history_user_emb = self.gnn_user_emb_buffer[batch['user_history_matrix']]
        item_history_emb = average_pooling(item_history_user_emb, batch['user_history_len']).detach()
        item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)

        # (*** 繼承 ***) 使用父類的 MLP 進行計算
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        
        # (*** 繼承 ***) 
        # EmbMLP 的 InfoNCE Loss 是 BCE，但 inference 卻用了 BCEWithLogits。
        # 為了與 EmbMLP.inference 保持一致，我們也使用 BCEWithLogits。
        # (雖然理論上用 pos_logits 計算 AUC/NDCG 即可)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits, batch['labels'].float(), reduction='none'
        )

        # --- 2. 計算負樣本的 Logits (使用 GNN Buffers) ---
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        # (*** MODIFIED ***) 
        # 關鍵修改：從 GNN BUFFER 獲取負樣本的靜態 Embedding
        neg_item_static_emb = self.gnn_item_emb_buffer[neg_item_ids_batch]
        
        # (*** MODIFIED ***)
        # 負樣本的 "物品歷史" 特徵：
        # 原始 EmbMLP 是從 self.item_history_buffer 讀取。
        # 但在 Hybrid 模型中，GNN buffer 已經包含了圖結構資訊 (隱含的用戶歷史)。
        # 因此，我們不再需要 self.item_history_buffer。
        # 我們需要的是負樣本的「用戶歷史特徵」(即 item_history_emb)
        neg_item_history_user_emb = self.gnn_user_emb_buffer[batch['user_history_matrix']]
        # 擴展 (B, L, D) -> (B, 1, L, D) -> (B, M, L, D)
        B, L, D = neg_item_history_user_emb.shape
        neg_item_history_user_emb_expanded = neg_item_history_user_emb.unsqueeze(1).expand(B, num_neg_samples, L, D)
        # (B, M, L)
        neg_user_history_len = batch['user_history_len'].unsqueeze(1).expand(B, num_neg_samples)
        
        # (B, M, D)
        neg_item_history_emb = average_pooling(neg_item_history_user_emb_expanded.reshape(-1, L, D), neg_user_history_len.reshape(-1))
        neg_item_history_emb = neg_item_history_emb.reshape(B, num_neg_samples, D).detach()

        # 負樣本的類別特徵
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len)
        
        # (B, M, D_feat)
        neg_item_features = torch.cat([
            neg_item_static_emb, 
            avg_cate_emb_for_neg_item,
            neg_item_history_emb # 使用 GNN-based 的用戶歷史
        ], dim=2)
        
        # (B, D_feat) -> (B, 1, D_feat) -> (B, M, D_feat)
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)

        # (*** 繼承 ***) 使用父類的 MLP 進行計算
        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)

        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        # --- 3. 返回結果 ---
        return pos_logits, neg_logits, per_sample_loss
    




# ===================================================================
# (*** NEW ***) LightGCN_Only (Baseline) 
# ===================================================================

class LightGCN_Only(nn.Module):
    """
    純 LightGCN 模型實作 (Baseline)。
    - 不使用任何序列特徵 (itemSeq, userSeq)
    - 不使用任何類別特徵 (cateId)
    - 不使用 Hybrid_GNN_MLP 中的後處理 MLP
    - 預測 = dot(GNN_User_Emb, GNN_Item_Emb)
    """
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 hyperparams: Dict, 
                 adj_matrix: torch.Tensor):
        
        super().__init__()
        
        # 1. 儲存 GNN 相關參數
        self.num_users = num_users
        self.num_items = num_items
        self.adj_matrix = adj_matrix # 儲存傳入的稀疏鄰接矩陣
        
        # 讀取 GNN 參數
        self.embed_dim = hyperparams.get('user_embed_dim', 64) # 假設 user/item 維度相同
        self.n_layers = hyperparams.get('n_layers', 2)
        
        print(f"[LightGCN_Only] Initialized with {self.n_layers} GNN layers and Embed Dim {self.embed_dim}.")

        # 2. 建立 GNN 所需的基礎 Embedding
        # 這些是唯一的可訓練參數
        self.user_emb_w = nn.Embedding(self.num_users, self.embed_dim)
        self.item_emb_w = nn.Embedding(self.num_items, self.embed_dim)
        
        # 3. 註冊 GNN 緩存 (Buffer)，用於 "評估"
        self.register_buffer(
            'gnn_user_emb_buffer', 
            torch.empty(self.num_users, self.embed_dim, device=self.user_emb_w.weight.device)
        )
        self.register_buffer(
            'gnn_item_emb_buffer', 
            torch.empty(self.num_items, self.embed_dim, device=self.item_emb_w.weight.device)
        )

        # 4. 初始化權重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb_w.weight)
        nn.init.xavier_uniform_(self.item_emb_w.weight)

    def _propagate_lightgcn(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (*** 繼承自 Hybrid ***)
        執行 LightGCN 的圖傳播 (保持梯度)。
        """
        user_emb_0 = self.user_emb_w.weight
        item_emb_0 = self.item_emb_w.weight
        
        all_emb = torch.cat([user_emb_0, item_emb_0], dim=0)
        embs_list = [all_emb] # 第 0 層

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs_list.append(all_emb)
            
        final_embs = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        
        final_user_embs, final_item_embs = torch.split(
            final_embs, [self.num_users, self.num_items]
        )
        
        return final_user_embs, final_item_embs

    def update_gnn_buffer(self):
        """
        (*** 繼承自 Hybrid ***)
        計算 GNN embedding 並將其 "凍結" 到 buffer 中，
        專門用於快速的 "評估" 階段。
        """
        print("--- (LightGCN_Only) Updating GNN buffer for evaluation ---")
        with torch.no_grad():
            user_embs, item_embs = self._propagate_lightgcn()
            self.gnn_user_emb_buffer.data.copy_(user_embs.detach())
            self.gnn_item_emb_buffer.data.copy_(item_embs.detach())

    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        """
        (*** 核心修改 ***)
        計算 GCN 的訓練損失。
        使用標準的 BCEWithLogitsLoss，因為 data_loader 提供了 0/1 標籤。
        """
        # --- 1. 執行 GNN 傳播 (保持梯度) ---
        gnn_user_embs, gnn_item_embs = self._propagate_lightgcn()

        # --- 2. 獲取當前批次的 Embedding ---
        u_emb = gnn_user_embs[batch['users']]
        i_emb = gnn_item_embs[batch['items']]
        
        # --- 3. 計算 Logits (純點積) ---
        logits = torch.sum(u_emb * i_emb, dim=1)
        
        # --- 4. 計算損失 ---
        labels = batch['labels'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss
        
    
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (*** 核心修改 ***)
        在 1-vs-M 評估中執行推論。
        """
        
        # --- 1. 從 GNN Buffer 獲取正樣本 Embedding ---
        # (B, D)
        u_emb = self.gnn_user_emb_buffer[batch['users']]
        # (B, D)
        pos_i_emb = self.gnn_item_emb_buffer[batch['items']]
        
        # --- 2. 計算正樣本 Logits ---
        # (B,)
        pos_logits = torch.sum(u_emb * pos_i_emb, dim=1)

        # --- 3. 從 GNN Buffer 獲取負樣本 Embedding ---
        # neg_item_ids_batch shape: (B, M)
        # neg_i_emb shape: (B, M, D)
        neg_i_emb = self.gnn_item_emb_buffer[neg_item_ids_batch]
        
        # --- 4. 計算負樣本 Logits ---
        # (B, D) -> (B, 1, D)
        u_emb_expanded = u_emb.unsqueeze(1)
        
        # (B, 1, D) * (B, M, D) -> sum(dim=2) -> (B, M)
        neg_logits = torch.sum(u_emb_expanded * neg_i_emb, dim=2)
        
        # --- 5. 計算損失 (用於 main.py 的指標，雖然 GNN 通常不用) ---
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits, batch['labels'].float(), reduction='none'
        )
        
        # --- 6. 返回結果 ---
        return pos_logits, neg_logits, per_sample_loss