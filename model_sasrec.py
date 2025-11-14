
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




# ===================================================================
# (*** NEW ***) PureSASRec (還原論文版)
# ===================================================================

class PureSASRec(nn.Module):
    """
    SASRec 論文的 "純粹" 實作 (Kang & McAuley, 2018)。
    
    - 結構: Session-based (單塔)，只使用 itemSeq。
    - 忽略: userId, userSeq, cateId。
    - Encoder: Transformer Encoder (含 Causal Mask)。
    - 聚合: 序列的 "最後一個" 輸出向量 (h_last)。
    - 預測: dot(h_last, all_item_embeddings)。
    - Loss: 繼承 main_model_utils.py 的 InfoNCE Loss。
    """
    
    def __init__(self, 
                 cates: np.ndarray, # (未使用)
                 cate_lens: np.ndarray, # (未使用)
                 hyperparams: Dict, 
                 train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        super().__init__()
        self.hparams = hyperparams
        
        # 1. 獲取核心參數
        self.num_items = self.hparams['num_items']
        self.embed_dim = self.hparams['item_embed_dim']
        self.maxlen = 30 # 來自 utils.py 的硬編碼
        
        # --- Transformer 超參數 ---
        n_heads = self.hparams.get('transformer_n_heads', 4)
        n_layers = self.hparams.get('transformer_n_layers', 2)
        dropout = self.hparams.get('transformer_dropout', 0.1)
        
        if self.embed_dim % n_heads != 0:
            n_heads = next(h for h in [4, 2, 1] if self.embed_dim % h == 0)
            print(f"Warning: item_embed_dim {self.embed_dim} not divisible by {n_heads}. Using {n_heads} heads.")

        # 2. 建立靜態 Item Embedding
        self.item_emb_w = nn.Embedding(self.num_items, self.embed_dim, padding_idx=0)
        # 3. 建立 Positional Embedding
        self.pos_emb_w = nn.Embedding(self.maxlen, self.embed_dim)
        
        # 4. 建立 Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=n_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            batch_first=True # (!!!)
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        
        # 5. LayerNorm (SASRec 論文使用)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-8)
        
        # 6. Loss 參數 (為了 InfoNCE)
        self.temperature = hyperparams.get('temperature', 0.07)

        self._init_weights()
        
    def _init_weights(self):
        # 論文的標準初始化
        nn.init.xavier_normal_(self.item_emb_w.weight)
        nn.init.xavier_normal_(self.pos_emb_w.weight)

    def _get_session_embedding(self, batch: Dict) -> torch.Tensor:
        """
        輔助函式：執行完整的 SASRec 流程 (Transformer + 聚合)
        
        Args:
            batch (Dict): 包含 'item_history_matrix' 和 'item_history_len' 的批次資料
        
        Returns:
            torch.Tensor: 代表批次中每個會話的 Embedding (B, D)
        """
        
        # --- 1. 獲取輸入和維度 ---
        # (B, T) - T 是 self.maxlen (例如 30)
        item_history_ids = batch['item_history_matrix']
        # (B,)
        seq_lens = batch['item_history_len']
        
        B, T = item_history_ids.shape
        D = self.embed_dim
        device = item_history_ids.device

        # --- 2. 準備 Embedding ---
        
        # (B, T, D)
        # 獲取序列中每個物品的靜態 Embedding
        # 注意: self.item_emb_w(padding_idx=0) 會自動將 padding (ID=0) 映射為 0 向量
        seq_emb = self.item_emb_w(item_history_ids)
        
        # 獲取位置 Embedding
        # (T,) -> (1, T, D)
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        pos_embeddings = self.pos_emb_w(pos_ids).unsqueeze(0) # (1, T, D)
        
        # --- 3. 建立輸入 (論文 Eq(1)) ---
        # (B, T, D)
        # 論文: E = [e_s1, ..., e_sn] + [p_1, ..., p_n]
        seq_emb_with_pos = seq_emb + pos_embeddings
        # 論文: 套用 LayerNorm 和 Dropout (Dropout 由 Transformer 模組處理)
        seq_emb_with_pos = self.layer_norm(seq_emb_with_pos) 
        
        # --- 4. 建立 Transformer Masks ---
        
        # Mask 1: Causal (Look-ahead) Mask (大小 T, T)
        # 這是 SASRec 的核心：位置 i 只能 attend 到 [0, ..., i]
        # True 代表 "不允許" attend
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device).bool()
        
        # Mask 2: Padding Mask (大小 B, T)
        # 標記出 padding (ID=0) 的位置
        # True 代表 "忽略"
        padding_mask = (item_history_ids == 0)
        
        # --- 5. 運行 Transformer Encoder (論文 Eq(2, 3)) ---
        
        # (B, T, D)
        transformer_output = self.transformer_encoder(
            src=seq_emb_with_pos,       # 輸入序列
            mask=causal_mask,           # Causal Mask (防止看到未來)
            src_key_padding_mask=padding_mask # Padding Mask (忽略 padding)
        )
        
        # --- 6. [!!! 關鍵修正：NaN 安全保護 !!!] ---
        #
        # 當 seq_lens == 0 時:
        # 1. padding_mask 會是 [True, True, ..., True] (全部遮罩)
        # 2. Transformer 內的 Softmax 會計算 softmax([-inf, -inf, ...])
        # 3. 這會導致 0 / 0 = NaN (Not a Number)
        # 4. transformer_output 充滿 NaN，導致後續 Loss 崩潰
        #
        # 我們的策略: 手動將 0 長度序列的輸出強制設為 0 向量
        
        # 1. 建立 0 長度遮罩 (B,) -> (B, 1, 1) -> (B, T, D)
        zero_len_mask = (seq_lens == 0).unsqueeze(-1).unsqueeze(-1).expand(B, T, D)
        
        # 2. 安全替換
        # 如果 zero_len_mask 為 True (即 seq_len == 0)，
        # 則使用 0.0，否則使用 transformer_output
        transformer_output = torch.where(
            zero_len_mask, 
            0.0, # 強制設為 0.0
            transformer_output
        )
        
        # 3. (額外保險) 將任何剩餘的 (理論上不應有) NaN 替換為 0
        transformer_output = torch.nan_to_num(transformer_output, nan=0.0)

        # --- 7. 聚合 (獲取最終會話 Embedding) ---
        
        # 論文: 只使用「最後一個」物品的輸出向量
        # (B,) -> (B, 1, 1)
        # (注意: clamp(min=0) 會安全地處理 seq_lens=0 的情況，使其索引為 (0-1).clamp=0)
        last_indices = (seq_lens - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, self.embed_dim)
        
        # (B, D)
        # 使用 torch.gather 根據索引從 (B, T, D) 中選取 (B, 1, D)
        h_session = torch.gather(transformer_output, 1, last_indices).squeeze(1) 
        
        # (對於 seq_len=0 的序列, h_session 現在會是 [0, 0, ..., 0], 這是安全的)
        
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
        h_session_norm = F.normalize(h_session, p=2, dim=-1, eps=1e-8)
        item_emb_norm = F.normalize(item_emb, p=2, dim=-1, eps=1e-8)
        
        return h_session_norm, item_emb_norm

    # (*** 複製 SR_GNN_Pure 的 InfoNCE Loss ***)
    def _calculate_infonce_loss(self, user_embedding, item_embedding, labels):
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
        session_embedding, item_embedding = self.forward(batch)
        losses = self._calculate_infonce_loss(session_embedding, item_embedding, batch['labels'])
        final_loss = losses.mean()
        return final_loss
        
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # --- 1. 計算 Session Embedding ---
        h_session = self._get_session_embedding(batch)
        h_session_norm = F.normalize(h_session, p=2, dim=-1, eps=1e-8)

        # --- 2. 獲取正樣本 Embedding ---
        pos_i_emb = self.item_emb_w(batch['items'])
        pos_i_emb_norm = F.normalize(pos_i_emb, p=2, dim=-1, eps=1e-8)
        
        # --- 3. 計算正樣本 Logits (Cosine Similarity) ---
        pos_logits = torch.sum(h_session_norm * pos_i_emb_norm, dim=1)

        # --- 4. 獲取負樣本 Embedding ---
        neg_i_emb = self.item_emb_w(neg_item_ids_batch)
        neg_i_emb_norm = F.normalize(neg_i_emb, p=2, dim=-1, eps=1e-8)
        
        # --- 5. 計算負樣本 Logits ---
        h_session_expanded = h_session_norm.unsqueeze(1)
        neg_logits = torch.sum(h_session_expanded * neg_i_emb_norm, dim=2)
        
        # --- 6. 計算 Loss (用於指標) ---
        pos_logits_for_loss = torch.sum(h_session * pos_i_emb, dim=1)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits_for_loss, batch['labels'].float(), reduction='none'
        )
        
        return pos_logits, neg_logits, per_sample_loss

# ===================================================================
# (*** NEW ***) EmbSASRec_Model (Transformer-based Encoder)
# ===================================================================

class SASRec_MLP(EmbMLP):
    """
    EmbMLP 的 Transformer (SASRec) 升級版。
    
    - 繼承 EmbMLP，重用 MLP 交互器、InfoNCE Loss 和雙塔架構。
    - 用 "Transformer Encoder + Masked AvgPool" 替換 "AvgPool"。
    - 建立兩套獨立的 Transformer (A 和 B)，以處理維度不同的 itemSeq 和 userSeq。
    """
    
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        # 1. 初始化父類 (EmbMLP)
        super().__init__(cates, cate_lens, hyperparams, train_config, 
                         item_init_vectors, cate_init_vectors)

        
        # --- 獲取維度 ---
        self.item_dim = self.hparams['item_embed_dim']
        self.user_dim = self.hparams['user_embed_dim']
        self.cate_dim = self.hparams['cate_embed_dim']
        
        # 兩個路徑的輸入維度
        self.item_seq_input_dim = self.item_dim + self.cate_dim
        self.user_seq_input_dim = self.user_dim
        
        # --- Transformer 超參數 (可從 config.yaml 讀取) ---
        self.maxlen = 30 # 來自 utils.py 的硬編碼
        n_heads = self.hparams.get('transformer_n_heads', 4)
        n_layers = self.hparams.get('transformer_n_layers', 2)
        dropout = self.hparams.get('transformer_dropout', 0.1)
        
        # --- 模組 A: 用於 itemSeq (Input: item_dim + cate_dim) ---
        dim_A = self.item_seq_input_dim
        # Positional Embedding (A)
        self.item_seq_pos_emb = nn.Embedding(self.maxlen, dim_A)
        # Transformer Encoder (A)
        # (注意: d_model 必須能被 n_heads 整除)
        if dim_A % n_heads != 0:
            # 找到一個能整除的 head 數量
            n_heads_A = next(h for h in [4, 2, 1] if dim_A % h == 0)
            print(f"Warning: item_seq_dim {dim_A} not divisible by {n_heads}. Using {n_heads_A} heads.")
        else:
            n_heads_A = n_heads
            
        transformer_layer_A = nn.TransformerEncoderLayer(
            d_model=dim_A,
            nhead=n_heads_A,
            dim_feedforward=dim_A * 4, # 慣例
            dropout=dropout,
            batch_first=True # (!!!)
        )
        self.item_seq_transformer = nn.TransformerEncoder(transformer_layer_A, num_layers=n_layers)

        # --- 模組 B: 用於 userSeq (Input: user_dim) ---
        dim_B = self.user_seq_input_dim
        # Positional Embedding (B)
        self.user_seq_pos_emb = nn.Embedding(self.maxlen, dim_B)
        # Transformer Encoder (B)
        if dim_B % n_heads != 0:
            n_heads_B = next(h for h in [4, 2, 1] if dim_B % h == 0)
            print(f"Warning: user_seq_dim {dim_B} not divisible by {n_heads}. Using {n_heads_B} heads.")
        else:
            n_heads_B = n_heads
            
        transformer_layer_B = nn.TransformerEncoderLayer(
            d_model=dim_B,
            nhead=n_heads_B,
            dim_feedforward=dim_B * 4,
            dropout=dropout,
            batch_first=True # (!!!)
        )
        self.user_seq_transformer = nn.TransformerEncoder(transformer_layer_B, num_layers=n_layers)


    def _run_transformer_encoder(self, 
                                 seq_emb: torch.Tensor, 
                                 seq_lens: torch.Tensor, 
                                 transformer_module: nn.TransformerEncoder, 
                                 pos_emb_module: nn.Embedding) -> torch.Tensor:
        """
        通用的 Transformer 執行器 + Masked Average Pooling
        """
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        # 1. 建立 Padding Mask (True 代表 "忽略")
        # (B, T)
        padding_mask = torch.arange(T, device=device)[None, :] >= seq_lens[:, None]

        # 2. 建立 Positional Embeddings
        # (T,) -> (1, T, D)
        pos_ids = torch.arange(T, dtype=torch.long, device=device)
        pos_embeddings = pos_emb_module(pos_ids).unsqueeze(0) # (1, T, D)
        
        # 3. 加入 Positional Embeddings
        seq_emb_with_pos = seq_emb + pos_embeddings
        
        # 4. 運行 Transformer
        # (不需要 causal mask，因為我們是做 "興趣包" 編碼，而非 "下一步" 預測)
        # (B, T, D)
        transformer_output = transformer_module(
            src=seq_emb_with_pos, 
            src_key_padding_mask=padding_mask
        )
        
        # 5. Masked Average Pooling (在 Transformer 的 *輸出* 上)
        # 這繼承了 EmbMLP 的魯棒性，但特徵是經過 self-attention 處理的
        
        # (B, T, D) -> (B, D)
        # 我們重用父類的 average_pooling 輔助函式 (!!!)
        # 注意：這個 average_pooling 函式在父類 EmbMLP 中
        pooled_output = average_pooling(transformer_output, seq_lens)
        
        return pooled_output


    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (*** 覆寫 EmbMLP ***)
        使用 v2.0 的 Transformer 模組取代 AvgPool
        """
        
        # === 使用者表示 (User Tower) ===
        static_u_emb = self.user_emb_w(batch['users']) # [B, user_dim]
        
        # 1. 準備 itemSeq (Transformer 的輸入)
        hist_item_emb = self.item_emb_w(batch['item_history_matrix']) # [B, T, item_dim]
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_len = self.cate_lens[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, hist_cates_len) # [B, T, cate_dim]
        
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2) # [B, T, item_seq_input_dim]

        # 2. 呼叫「模組 A」
        user_history_emb = self._run_transformer_encoder(
            hist_item_emb_with_cate, batch['item_history_len'],
            self.item_seq_transformer, self.item_seq_pos_emb
        ) # [B, item_seq_input_dim]
        
        user_features = torch.cat([static_u_emb, user_history_emb.detach()], dim=-1)
        # user_features = torch.cat([static_u_emb, user_history_emb], dim=-1)

        # === 物品表示 (Item Tower) ===
        static_item_emb = self.item_emb_w(batch['items']) # [B, item_dim]
        item_cates = self.cates[batch['items']]
        item_cates_len = self.cate_lens[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, item_cates_len) # [B, cate_dim]
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1) # [B, item_dim + cate_dim]
        
        # 1. 準備 userSeq (Transformer 的輸入)
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix']) # [B, T, user_seq_input_dim]

        # 2. 呼叫「模組 B」
        item_history_emb = self._run_transformer_encoder(
            item_history_user_emb, batch['user_history_len'],
            self.user_seq_transformer, self.user_seq_pos_emb
        ) # [B, user_seq_input_dim]
        
        item_features = torch.cat([item_emb_with_cate, item_history_emb.detach()], dim=-1)
        # item_features = torch.cat([item_emb_with_cate, item_history_emb], dim=-1)


        # 1. 從 item_features 中分離出我們想要快取的 "userSeq 編碼結果"
        # (item_features = [item_emb_with_cate, item_history_emb])
        item_history_emb = item_features[:, -self.user_seq_input_dim:] # [B, user_dim]

        # 2. 在訓練時，更新快取
        if self.training:
            self.user_history_buffer.weight[batch['items']] = item_history_emb.detach()
        
        return user_features, item_features

    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # --- 1. 計算正樣本的 Embedding 和 Logits ---
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits, batch['labels'].float(), reduction='none'
        )

        # --- 2. 計算負樣本的 Logits ---
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        # 負樣本的靜態部分
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch) # [B, M, item_dim]
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        neg_item_cates_len = self.cate_lens[neg_item_ids_batch]
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, neg_item_cates_len) # [B, M, cate_dim]
        neg_item_emb_with_cate = torch.cat([neg_item_static_emb, avg_cate_emb_for_neg_item], dim=2) # [B, M, item_dim + cate_dim]
        
        # 負樣本的歷史部分 (userSeq)
        item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        
        
        # 組合負樣本特徵
        neg_item_features = torch.cat([
            neg_item_emb_with_cate, 
            item_history_emb_expanded.detach() # .detach() 確保安全
        ], dim=2)

        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)

        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss