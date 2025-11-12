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


class LightGCN(nn.Module):
    """
    LightGCN 模型的 PyTorch 實現。
    """
    def __init__(self, num_users: int, num_items: int, hyperparams: Dict,
                 adj_matrix: torch.sparse.FloatTensor):
        """
        Args:
            num_users (int): 用戶總數
            num_items (int): 物品總數
            hyperparams (Dict): 包含 'embed_dim' 和 'num_layers' 的字典
            adj_matrix (torch.sparse.FloatTensor): 
                (N_u + N_i) x (N_u + N_i) 的「對稱正規化」鄰接矩陣
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = hyperparams.get('embed_dim', 64)
        self.num_layers = hyperparams.get('num_layers', 3)
        self.hparams = hyperparams
        
        print(f"--- Initializing LightGCN ---")
        print(f"  Users: {self.num_users}, Items: {self.num_items}")
        print(f"  Embed Dim: {self.embed_dim}, Layers: {self.num_layers}")
        
        # 1. 唯一的「可訓練」參數：E_0
        self.embeddings = nn.Embedding(
            num_users + num_items, 
            self.embed_dim
        )
        # (我們也把 user_history_buffer 留著，以防萬一，雖然 GNN 不太用它)
        self.user_history_buffer = nn.Embedding(num_users, self.embed_dim)

        # 2. 註冊「圖」 (它不是參數，是 buffer)
        self.register_buffer('adj_matrix', adj_matrix)

        self._init_weights()

    def _init_weights(self):
        # LightGCN 使用 Xavier 初始化 E_0
        nn.init.xavier_uniform_(self.embeddings.weight)
        print("--- LightGCN Embeddings initialized (Xavier) ---")

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        執行 LightGCN 的 K 層圖卷積 (Graph Convolution)。
        
        這是在「全圖」上操作的，它*沒有* batch 參數。
        它在訓練迴圈中每個 epoch 只被呼叫一次。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                (users_final_emb, items_final_emb)
        """
        
        # E_k: 儲存每一層的 Embedding
        all_embeddings = self.embeddings.weight
        embeddings_k = [all_embeddings] # 第 0 層 (E_0)
        
        # --- GNN 傳播 (K 層) ---
        for layer in range(self.num_layers):
            # 核心邏輯: E_{k+1} = (D^{-1/2} A D^{-1/2}) @ E_k
            # all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            all_embeddings = spmm(
                self.adj_matrix.indices(), 
                self.adj_matrix.values(), 
                self.adj_matrix.shape[0], 
                self.adj_matrix.shape[1], 
                all_embeddings
            )
            embeddings_k.append(all_embeddings)

        # --- 最終輸出 (LightGCN 的秘方) ---
        # 最終的 Embedding E_final 是 K+1 層的「加權平均」
        # (標準 LightGCN 是「平均」，我們這裡用 1/(K+1))
        final_embeddings = torch.mean(torch.stack(embeddings_k, dim=0), dim=0)
        
        # 切分回 user 和 item
        users_emb, items_emb = torch.split(
            final_embeddings, 
            [self.num_users, self.num_items]
        )
        
        # 在訓練時，順便更新一下 history buffer (為了舊的評估邏輯)
        if self.training:
            self.user_history_buffer.weight.data = users_emb.detach()
        
        return users_emb, items_emb

    def calculate_loss(self, users_emb: torch.Tensor, items_emb: torch.Tensor, 
                       batch: Dict) -> torch.Tensor:
        """
        計算 LightGCN 的 BPR Loss (Bayesian Personalized Ranking)。
        
        Args:
            users_emb (torch.Tensor): 來自 forward() 的「最終」用戶向量
            items_emb (torch.Tensor): 來自 forward() 的「最終」物品向量
            batch (Dict): 來自 BPRDataset 的批次，包含 (u, i, j)
        """
        
        # 1. 獲取 BPR 批次
        user_ids = batch['user_id']
        pos_item_ids = batch['pos_item_id']
        neg_item_ids = batch['neg_item_id']

        # 2. 查找 GNN 產生的最終 embedding
        u_emb = users_emb[user_ids]
        pos_i_emb = items_emb[pos_item_ids]
        neg_i_emb = items_emb[neg_item_ids]

        # 3. BPR Loss: -log(sigmoid(pos_score - neg_score))
        pos_score = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_score = torch.sum(u_emb * neg_i_emb, dim=1)
        
        # F.softplus(x) = log(1 + exp(x))
        # BPR Loss = mean( -log( sigmoid(pos - neg) ) )
        #          = mean( -log( 1 / (1 + exp(neg - pos)) ) )
        #          = mean( log( 1 + exp(neg - pos) ) )
        #          = mean( softplus( neg - pos ) )
        loss = F.softplus(neg_score - pos_score).mean()
        
        # (可選) 加上 L2 正規化 (Weight Decay)
        l2_reg = self.hparams.get('l2_reg', 1e-4)
        if l2_reg > 0:
            # 只對「當前批次」用到的 E_0 向量進行正規化
            # (這是 LightGCN 論文的推薦作法)
            u_emb_0 = self.embeddings(user_ids)
            pos_i_emb_0 = self.embeddings(pos_item_ids + self.num_users)
            neg_i_emb_0 = self.embeddings(neg_item_ids + self.num_users)
            
            reg_loss = (u_emb_0.norm(2).pow(2) + 
                        pos_i_emb_0.norm(2).pow(2) + 
                        neg_i_emb_0.norm(2).pow(2)) / float(len(user_ids))
            
            loss = loss + l2_reg * reg_loss

        return loss

    
    def inference_for_evaluation(self, 
                                 users_emb: torch.Tensor, 
                                 items_emb: torch.Tensor,
                                 batch: Dict, 
                                 neg_user_ids_batch: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (*** 新函式 ***)
        為了和你「舊的」評估標準 (99 neg_users vs 1 pos_item) 保持一致。
        """
        
        # a. 獲取「正樣本」向量
        # (B, D)
        pos_user_emb = users_emb[batch['users']]
        # (B, D)
        pos_item_emb = items_emb[batch['items']]

        # b. 獲取「負樣本」向量
        # (B, M, D)
        neg_user_emb = users_emb[neg_user_ids_batch]
        
        # c. 計算分數
        # 正樣本分數 (u, i)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1) # (B,)
        
        # 負樣本分數 (neg_u, i)
        # 擴展 item 向量: (B, D) -> (B, 1, D)
        neg_logits = torch.sum(neg_user_emb * pos_item_emb.unsqueeze(1), dim=2) # (B, M)
        
        return pos_logits, neg_logits