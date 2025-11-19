
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
    
    def calculate_loss(self, batch: Dict, teacher_model: torch.nn.Module = None, kd_weight: float = 0.0) -> torch.Tensor:
        """
        (*** MODIFIED ***) 加入 Teacher Model 和 kd_weight 參數
        """
        # 1. Student (當前模型) 的輸出 (已 L2 歸一化)
        student_session_emb, student_item_emb = self.forward(batch)
        
        # 2. 基礎推薦 Loss (InfoNCE)
        loss_infonce = self._calculate_infonce_loss(student_session_emb, student_item_emb, batch['labels'])
        loss_main = loss_infonce.mean()
        
        # 3. [!!! NEW !!!] 自我蒸餾 (Self-Distillation)
        loss_kd = 0.0
        if teacher_model is not None and kd_weight > 0:
            with torch.no_grad():
                # 獲取 Teacher 對「同一批資料」的 Embedding
                # 注意：Teacher 是凍結的，不需要梯度
                teacher_session_emb, teacher_item_emb = teacher_model.forward(batch)
            
            # 計算 MSE Loss (Feature-based Distillation)
            # 強迫 Student 的 Embedding 空間不要偏離 Teacher 太遠
            loss_user_kd = F.mse_loss(student_session_emb, teacher_session_emb)
            loss_item_kd = F.mse_loss(student_item_emb, teacher_item_emb)
            
            loss_kd = (loss_user_kd + loss_item_kd) * kd_weight

        return loss_main + loss_kd

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
    


class CausalSASRec_MLP(SASRec_MLP):
    """
    因果去偏版 SASRec (Causal Debiasing).
    
    架構:
    1. Main Branch: 繼承 SASRec_MLP (Transformer + MLP)，學習 User-Item 匹配。
    2. Bias Branch: 一個簡單的 Item Bias Embedding，學習物品的流行度偏差。
    
    訓練時: Score = Main_Score + Item_Bias
    推論時: Score = Main_Score (直接去除偏差)
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        # 1. 初始化主分支 (EmbSASRec)
        super().__init__(cates, cate_lens, hyperparams, train_config, 
                         item_init_vectors, cate_init_vectors)
        
        print("[CausalSASRec_MLP] Initialized. Adding Bias Branch for Causal Debiasing.")
        
        # 2. 建立偏差分支 (Bias Tower)
        # 這只是一個純量 (Scalar)，代表物品的固有吸引力 (流行度)
        # 初始化為 0，讓模型從頭學起
        self.item_bias_emb = nn.Embedding(self.hparams['num_items'], 1)
        nn.init.zeros_(self.item_bias_emb.weight)
        
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (*** 覆寫 ***) 訓練前向傳播
        Returns:
            user_emb, item_emb (來自 Main Branch)
            item_bias (來自 Bias Branch)
        """
        # 1. 執行主分支 (Main Branch) - 繼承自 EmbSASRec
        # 這包含了 Transformer 編碼和 MLP 交互
        user_emb, item_emb = super().forward(batch)
        
        # 2. 執行偏差分支 (Bias Branch)
        # 獲取當前 Batch 內所有物品的 Bias 值
        # (B,) -> (B, 1)
        item_bias = self.item_bias_emb(batch['items'])
        
        return user_emb, item_emb, item_bias

    def calculate_loss(self, batch: Dict) -> torch.Tensor:
        """
        計算因果去偏的 InfoNCE Loss
        """
        # 1. 獲取輸出
        user_emb, item_emb, pos_item_bias = self.forward(batch)
        
        # 2. 計算 Loss (加入 Bias)
        # 我們需要重寫 InfoNCE 的邏輯來加入 Bias
        loss = self._calculate_causal_infonce_loss(user_emb, item_emb, pos_item_bias, batch)
        
        return loss

    def _calculate_causal_infonce_loss(self, user_embedding, item_embedding, pos_item_bias, batch):
        """
        因果 InfoNCE Loss:
        Logits = (User @ Item.T) / temp + Bias.T
        """
        # --- Main Branch Logits ---
        # (B, D) @ (D, B) -> (B, B)
        # main_logits[i, j] = User_i 對 Item_j 的興趣分數
        main_logits = torch.matmul(user_embedding, item_embedding.t()) / self.temperature
        
        # --- Bias Branch Logits ---
        # 我們需要 Batch 內「所有」物品的 Bias
        # pos_item_bias 是 (B, 1)，對應 batch['items'] 的 bias
        
        # 關鍵：對於 InfoNCE 的每一對 (User_i, Item_j)，
        # 我們都要加上 Item_j 的 bias。
        # 所以我們將 (B, 1) 的 bias 轉置為 (1, B)，然後廣播加到 (B, B) 上
        all_item_biases = pos_item_bias.t() # (1, B)
        
        # --- 融合 (Fusion) ---
        # Final_Logits[i, j] = Main_Score(u_i, i_j) + Bias(i_j)
        # 這樣，如果 Item_j 很熱門 (Bias 高)，Main_Score 就可以小一點
        # Main_Score 被迫學習 "扣除熱門度後" 的殘差
        final_logits = main_logits + all_item_biases
        
        # --- 標準 InfoNCE 計算 (使用 final_logits) ---
        logits_max, _ = torch.max(final_logits, dim=1, keepdim=True)
        logits_stabilized = final_logits - logits_max
        exp_logits_den = torch.exp(logits_stabilized)
        denominator = exp_logits_den.sum(dim=1, keepdim=True)
        
        # 分子 (正樣本對)
        # Pos[i] = Final_Logits[i, i] = Main[i,i] + Bias[i]
        pred_logits = torch.diag(final_logits).unsqueeze(1) # (B, 1)
        
        pred_logits_stabilized = pred_logits - logits_max
        numerator = torch.exp(pred_logits_stabilized)

        # .squeeze(dim=1) 修復 batch_size=1 的 bug
        infonce_pred = (numerator / (denominator + 1e-9)).squeeze(dim=1)
        infonce_pred = torch.clamp(infonce_pred, min=1e-9, max=1.0 - 1e-9)
        
        return F.binary_cross_entropy(infonce_pred, batch['labels'].float(), reduction='none').mean()


    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor,
        bias_strategy: str = 'mainstream', # 'weighted' or 'mainstream'
        bias_weight: float = 0.3         # 用於 weighted 策略的 lambda
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. 獲取 Main Branch 的 logits (User-Item 匹配分)
        # 呼叫父類 (EmbSASRec) 的 inference 邏輯，這會給出純粹的 Score_main
        pos_logits_main, neg_logits_main, _ = super().inference(batch, neg_item_ids_batch)
        
        # 2. 獲取 Bias Branch 的 logits (物品流行度分)
        # 正樣本 Bias
        pos_item_bias = self.item_bias_emb(batch['items']).squeeze() # (B,)
        # 負樣本 Bias
        neg_item_bias = self.item_bias_emb(neg_item_ids_batch).squeeze() # (B, M)
        
        # 3. 策略選擇與組合
        
        if bias_strategy == 'weighted':
            # --- 策略一：靜態加權 (Soft Intervention) ---
            # Score = Main + lambda * Bias
            # bias_weight 建議 0.1 ~ 0.5
            pos_logits_final = pos_logits_main + bias_weight * pos_item_bias
            neg_logits_final = neg_logits_main + bias_weight * neg_item_bias
            
        elif bias_strategy == 'mainstream':
            # --- 策略二：用戶主流度感知 (User Mainstreaminess) ---
            # w_u = mean(Bias(History))
            
            # 1. 取出用戶歷史物品的 ID
            # batch['item_history_matrix']: (B, T)
            hist_ids = batch['item_history_matrix']
            
            # 2. 查表獲取歷史物品的 Bias
            # (B, T) -> (B, T, 1) -> (B, T)
            hist_bias = self.item_bias_emb(hist_ids).squeeze(-1)
            
            # 3. 計算平均 Bias (忽略 padding 0)
            mask = (hist_ids != 0).float()
            sum_bias = (hist_bias * mask).sum(dim=1)
            count = mask.sum(dim=1) + 1e-9
            avg_bias = sum_bias / count # (B,) 代表用戶的平均主流度
            
            # 4. 將 avg_bias 歸一化或通過 Sigmoid 作為權重
            # 這裡假設 Bias 可能是負無窮到正無窮，用 Sigmoid 壓到 (0, 1)
            # * 2.0 是一個 scaling factor，讓權重更有彈性
            user_weights = torch.sigmoid(avg_bias).unsqueeze(1) # (B, 1)
            
            # 5. 組合
            # Score = Main + w_u * Bias
            pos_logits_final = pos_logits_main + user_weights.squeeze() * pos_item_bias
            neg_logits_final = neg_logits_main + user_weights * neg_item_bias

        else:
            # 預設：不使用 Bias (即 lambda=0)
            pos_logits_final = pos_logits_main
            neg_logits_final = neg_logits_main

        
        # 4. 計算用於指標的 Loss (這裡只是一個參考，不影響排名)
        # 為了計算方便，這裡還是用 Main Logits 算 Loss，或者您也可以用 Final Logits
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits_final, batch['labels'].float(), reduction='none'
        )

        return pos_logits_final, neg_logits_final, per_sample_loss
    


# model_sasrec.py

class ContextSASRec_MLP(SASRec_MLP):
    """
    全域上下文 Prompt SASRec (Global Context as Prompt).
    
    核心機制:
    1. 計算全域 Context (Item Embedding 平均 + EMA)。
    2. 將 Context 投影到 Transformer 的維度。
    3. 將 Context 作為第 0 個 Token (Prompt) 拼接到序列前端: [Context, Item_1, Item_2...]。
    4. 透過 Self-Attention 讓歷史物品與當下環境互動。
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        # 1. 初始化父類 (EmbSASRec_Model)
        super().__init__(cates, cate_lens, hyperparams, train_config, 
                         item_init_vectors, cate_init_vectors)
        
        print("[ContextPromptSASRec_MLP] Initialized. Injecting Context as Transformer Prompt.")
        
        # 2. Context 來源維度
        self.context_source_dim = self.hparams['item_embed_dim']
        
        # 3. 投影層
        self.transformer_dim_A = self.item_seq_input_dim
        self.context_proj_A = nn.Linear(self.context_source_dim, self.transformer_dim_A)
        
        self.transformer_dim_B = self.user_seq_input_dim
        self.context_proj_B = nn.Linear(self.context_source_dim, self.transformer_dim_B)
        
        # 4. 全域環境向量快取
        self.register_buffer('global_context_ema', torch.zeros(self.context_source_dim))
        self.ema_alpha = 0.95 

        # [!!! 關鍵修正：擴大 Positional Embedding !!!]
        # 父類只分配了 maxlen (例如 30) 的空間。
        # 我們加入了 Prompt，序列長度會變成 maxlen + 1。
        # 所以我們需要重新初始化 Pos Emb，容量設為 maxlen + 1。
        
        new_maxlen = self.maxlen + 1
        
        # 覆蓋模組 A (itemSeq) 的 Pos Emb
        self.item_seq_pos_emb = nn.Embedding(new_maxlen, self.transformer_dim_A)
        
        # 覆蓋模組 B (userSeq) 的 Pos Emb
        self.user_seq_pos_emb = nn.Embedding(new_maxlen, self.transformer_dim_B)
        
        # 初始化權重 (保持與父類一致的初始化方式)
        nn.init.xavier_normal_(self.item_seq_pos_emb.weight)
        nn.init.xavier_normal_(self.user_seq_pos_emb.weight)

    def _update_context_ema(self, current_batch_context: torch.Tensor):
        """ 更新 EMA (同前) """
        
        self.global_context_ema.data.mul_(self.ema_alpha).add_(
            current_batch_context.detach(), alpha=(1.0 - self.ema_alpha)
        )

    def _run_transformer_with_prompt(self, 
                                     seq_emb: torch.Tensor, 
                                     seq_lens: torch.Tensor, 
                                     prompt_emb: torch.Tensor,
                                     transformer_module: nn.TransformerEncoder, 
                                     pos_emb_module: nn.Embedding) -> torch.Tensor:
        """
        帶 Prompt 的 Transformer 執行器
        """
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        # --- 1. 拼接 Prompt ---
        # prompt_emb: (B, D) -> (B, 1, D)
        prompt_emb = prompt_emb.unsqueeze(1)
        
        # 新序列: [Prompt, Item_1, Item_2, ..., Item_T]
        # (B, T+1, D)
        seq_emb_with_prompt = torch.cat([prompt_emb, seq_emb], dim=1)
        new_T = T + 1
        
        # --- 2. 處理 Mask ---
        # 原始 Padding Mask: (B, T)
        # 新 Padding Mask: (B, T+1) -> 第 0 位 (Prompt) 永遠不是 Padding (False)
        original_padding_mask = torch.arange(T, device=device)[None, :] >= seq_lens[:, None]
        prompt_padding_mask = torch.zeros(B, 1, device=device, dtype=torch.bool) # False
        new_padding_mask = torch.cat([prompt_padding_mask, original_padding_mask], dim=1)

        # --- 3. 位置編碼 ---
        # (T+1,)
        pos_ids = torch.arange(new_T, dtype=torch.long, device=device)
        pos_embeddings = pos_emb_module(pos_ids).unsqueeze(0) # (1, T+1, D)
        
        # 加入位置編碼
        seq_input = seq_emb_with_prompt + pos_embeddings
        
        # --- 4. 運行 Transformer ---
        # (不需要 Causal Mask ? 其實還是需要，保持自回歸特性，雖然這裡是編碼器)
        # 但因為我們只取最後一個，且 item_t 不應該看到 item_t+1
        # 所以我們還是加上 Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(new_T, device=device).bool()
        
        transformer_output = transformer_module(
            src=seq_input,
            mask=causal_mask,
            src_key_padding_mask=new_padding_mask
        )
        
        # --- 5. 聚合 (取最後一個 "有效" 物品) ---
        # 注意索引變化：
        # 原本 Item_1 在 index 0。現在 Item_1 在 index 1。
        # 原本最後一個物品在 index = seq_len - 1。
        # 現在最後一個物品在 index = (seq_len - 1) + 1 = seq_len。
        
        # 處理空序列 (seq_len=0) 的情況：
        # 如果 seq_len=0，我們希望取 index 0 (Prompt 本身) 或是 0 向量？
        # 取 Prompt 本身比較合理，代表"沒有歷史，只有環境"。
        
        # (B, 1, 1)
        target_indices = seq_lens.view(B, 1, 1).expand(-1, 1, D)
        
        # (B, D)
        pooled_output = torch.gather(transformer_output, 1, target_indices).squeeze(1)
        
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (*** 覆寫 ***) 注入 Context Prompt
        """
        # 1. 準備 Context Vector
        current_items_static = self.item_emb_w(batch['items'])
        batch_context = current_items_static.mean(dim=0) # (Context_Dim,)
        
        if self.training:
            self._update_context_ema(batch_context)
            context_to_use = batch_context
        else:
            context_to_use = self.global_context_ema
            
        # 擴展到 Batch: (B, Context_Dim)
        context_batch = context_to_use.unsqueeze(0).expand(len(batch['users']), -1)
        
        # 2. 準備 Prompt for A and B
        prompt_A = self.context_proj_A(context_batch) # (B, dim_A)
        prompt_B = self.context_proj_B(context_batch) # (B, dim_B)
        
        # === 使用者表示 (User Tower) ===
        static_u_emb = self.user_emb_w(batch['users'])
        
        # 準備 itemSeq
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, self.cate_lens[batch['item_history_matrix']])
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)

        # [!!! 呼叫帶 Prompt 的 Transformer !!!]
        user_history_emb = self._run_transformer_with_prompt(
            hist_item_emb_with_cate, batch['item_history_len'], prompt_A,
            self.item_seq_transformer, self.item_seq_pos_emb
        )
        
        user_features = torch.cat([static_u_emb, user_history_emb.detach()], dim=-1)

        # === 物品表示 (Item Tower) ===
        static_item_emb = self.item_emb_w(batch['items'])
        item_cates = self.cates[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, self.cate_lens[batch['items']])
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        
        # 準備 userSeq
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix'])

        # [!!! 呼叫帶 Prompt 的 Transformer !!!]
        item_history_emb = self._run_transformer_with_prompt(
            item_history_user_emb, batch['user_history_len'], prompt_B,
            self.user_seq_transformer, self.user_seq_pos_emb
        )
        
        item_features = torch.cat([item_emb_with_cate, item_history_emb.detach()], dim=-1)
        
        return user_features, item_features

    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        neg_item_ids_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推論 (需要手動複製邏輯以注入 Prompt)
        """
        # 1. 正樣本與特徵 (使用上面的 _build_feature_representations 即可)
        user_features, item_features = self._build_feature_representations(batch)
        pos_user_emb, pos_item_emb = self._get_embeddings_from_features(user_features, item_features)
        pos_logits = torch.sum(pos_user_emb * pos_item_emb, dim=1)
        
        per_sample_loss = F.binary_cross_entropy_with_logits(
            pos_logits, batch['labels'].float(), reduction='none'
        )

        # 2. 負樣本 (這部分比較麻煩，需要手動注入 Prompt 到負樣本的歷史計算中)
        num_neg_samples = neg_item_ids_batch.shape[1]
        
        # 準備 Context (推論用 EMA)
        context_to_use = self.global_context_ema
        context_batch = context_to_use.unsqueeze(0).expand(len(batch['users']), -1)
        # 注意：這裡負樣本計算用到的是 userSeq (物品的歷史)，所以用 prompt_B
        prompt_B = self.context_proj_B(context_batch) # (B, dim_B)
        
        # ... (負樣本靜態特徵部分同 EmbSASRec) ...
        neg_item_static_emb = self.item_emb_w(neg_item_ids_batch)
        neg_item_cate_emb = self.cate_emb_w(self.cates[neg_item_ids_batch])
        avg_cate_emb_for_neg_item = average_pooling(neg_item_cate_emb, self.cate_lens[neg_item_ids_batch])
        neg_item_emb_with_cate = torch.cat([neg_item_static_emb, avg_cate_emb_for_neg_item], dim=2)

        # ... (負樣本歷史部分) ...
        # 這裡我們需要重新計算嗎？
        # 我們的架構是雙塔，負樣本的 userSeq (物品歷史) 應該是從 batch['user_history_matrix'] 取出的嗎？
        # 不，負樣本是隨機採樣的物品，它們有自己的 userSeq。
        # 但 EmbMLP/SASRec 的 inference 簡化邏輯是假設我們無法即時獲取負樣本的 userSeq，
        # 而是重用正樣本的 userSeq (item_history_emb) 或者使用快取。
        
        # [重要] 我們延續 EmbSASRec v2.1 的快取邏輯
        # 我們在 _build_feature_representations 中已經算好了帶 Prompt 的 item_history_emb (userSeq result)
        # 並存入了快取 (如果啟用)。
        
        # 讓我們檢查父類的 cache 機制。
        # EmbSASRec v2.1 有 item_history_buffer。
        # _build_feature_representations 會把算好的 (包含 Prompt 影響的) embedding 存進去。
        # 所以 inference 只需要讀取 cache 即可，不需要再手動加 Prompt！
        
        # 直接讀取 Cache
        if hasattr(self, 'user_history_buffer'):
             item_history_emb_expanded = self.user_history_buffer(neg_item_ids_batch)
        else:
             # 如果沒有 cache (例如第一輪)，退化為重用正樣本特徵 (雖然不太正確但可跑)
             item_history_emb_dim = self.user_seq_input_dim
             item_history_emb = item_features[:, -item_history_emb_dim:]
             item_history_emb_expanded = item_history_emb.unsqueeze(1).expand(-1, num_neg_samples, -1)

        neg_item_features = torch.cat([neg_item_emb_with_cate, item_history_emb_expanded.detach()], dim=2)
        
        user_features_expanded = user_features.unsqueeze(1).expand(-1, num_neg_samples, -1)
        neg_user_emb_final, neg_item_emb_final = self._get_embeddings_from_features(user_features_expanded, neg_item_features)
        
        neg_logits = torch.sum(neg_user_emb_final * neg_item_emb_final, dim=2)
        
        return pos_logits, neg_logits, per_sample_loss
    



class DualPromptSASRec(ContextSASRec_MLP):
    """
    雙重提示 SASRec (Dual Prompt: Learnable + Context).
    
    輸入序列結構:
    [Learnable_Prompt, Context_Prompt, Item_1, Item_2, ..., Item_T]
    
    - Learnable_Prompt: 學習全域、長期的隱式模式 (Implicit Prior).
    - Context_Prompt: 注入當下、顯式的環境資訊 (Explicit Context).
    """
    def __init__(self, cates: np.ndarray, cate_lens: np.ndarray, hyperparams: Dict, train_config: Dict,
                 item_init_vectors: torch.Tensor = None,  
                 cate_init_vectors: torch.Tensor = None): 
        
        # 1. 初始化父類 (ContextPromptSASRec_MLP)
        # 這會建立 Context Projection Layer, EMA Buffer 等
        super().__init__(cates, cate_lens, hyperparams, train_config, 
                         item_init_vectors, cate_init_vectors)
        
        print("[DualPromptSASRec] Initialized. Using Dual Prompts (Learnable + Context).")
        
        # 2. 定義可學習的 Prompt (Learnable Parameters)
        # 針對 itemSeq (模組 A)
        self.learnable_prompt_A = nn.Parameter(torch.randn(1, self.transformer_dim_A))
        nn.init.xavier_normal_(self.learnable_prompt_A)
        
        # 針對 userSeq (模組 B)
        self.learnable_prompt_B = nn.Parameter(torch.randn(1, self.transformer_dim_B))
        nn.init.xavier_normal_(self.learnable_prompt_B)
        
        # 3. [!!! 關鍵修正 !!!] 再次擴大 Positional Embedding
        # 父類將其設為 maxlen + 1 (1個 Prompt)
        # 我們現在有 2 個 Prompt，所以需要 maxlen + 2
        
        new_maxlen = self.maxlen + 2
        
        # 覆蓋模組 A 的 Pos Emb
        self.item_seq_pos_emb = nn.Embedding(new_maxlen, self.transformer_dim_A)
        nn.init.xavier_normal_(self.item_seq_pos_emb.weight)
        
        # 覆蓋模組 B 的 Pos Emb
        self.user_seq_pos_emb = nn.Embedding(new_maxlen, self.transformer_dim_B)
        nn.init.xavier_normal_(self.user_seq_pos_emb.weight)
        
        print(f"   - Resized Pos Embeddings to {new_maxlen} to accommodate 2 Prompt tokens.")

    def _run_transformer_with_prompt(self, 
                                     seq_emb: torch.Tensor, 
                                     seq_lens: torch.Tensor, 
                                     context_prompt: torch.Tensor, # 來自 EMA
                                     learnable_prompt: torch.Tensor, # 來自 Parameter
                                     transformer_module: nn.TransformerEncoder, 
                                     pos_emb_module: nn.Embedding) -> torch.Tensor:
        """
        帶雙重 Prompt 的 Transformer 執行器
        """
        B, T, D = seq_emb.shape
        device = seq_emb.device
        
        # --- 1. 準備 Prompts ---
        # context_prompt: (B, D) -> (B, 1, D)
        context_prompt = context_prompt.unsqueeze(1)
        
        # learnable_prompt: (1, D) -> (B, 1, D)
        learnable_prompt = learnable_prompt.unsqueeze(0).expand(B, -1, -1)
        
        # --- 2. 拼接序列 ---
        # 順序: [Learnable, Context, History...]
        # 讓 Learnable 在最前面，作為一種全域的 [CLS] token 感覺
        # (B, T+2, D)
        seq_emb_with_prompts = torch.cat([learnable_prompt, context_prompt, seq_emb], dim=1)
        new_T = T + 2
        
        # --- 3. 處理 Mask ---
        # 原始 Padding Mask: (B, T)
        # 新 Padding Mask: (B, T+2) -> 前 2 位永遠不是 Padding (False)
        prompt_padding_mask = torch.zeros(B, 2, device=device, dtype=torch.bool) 
        
        # (注意: 這裡假設 seq_lens 是原始歷史長度，不包含 prompt)
        original_padding_mask = torch.arange(T, device=device)[None, :] >= seq_lens[:, None]
        
        new_padding_mask = torch.cat([prompt_padding_mask, original_padding_mask], dim=1)

        # --- 4. 位置編碼 ---
        pos_ids = torch.arange(new_T, dtype=torch.long, device=device)
        pos_embeddings = pos_emb_module(pos_ids).unsqueeze(0)
        seq_input = seq_emb_with_prompts + pos_embeddings
        
        # --- 5. 運行 Transformer ---
        # Causal Mask 大小為 T+2
        causal_mask = nn.Transformer.generate_square_subsequent_mask(new_T, device=device).bool()
        
        transformer_output = transformer_module(
            src=seq_input,
            mask=causal_mask,
            src_key_padding_mask=new_padding_mask
        )
        
        # --- 6. [!!! 關鍵修正 !!!] NaN 安全保護 ---
        # 如果 seq_len == 0，original_padding_mask 全為 True。
        # 但因為我們加了 2 個 Prompt，new_padding_mask 前兩位是 False。
        # 所以 softmax 不會全遮罩，不會產生 NaN！
        # transformer_output 的前兩位會有值 (Prompt 的 Self-Attention)。
        # 但是，後面的歷史部分仍然被遮罩。
        
        # 我們需要取最後一個「有效」物品。
        # 如果 seq_len > 0: 取 item_history 的最後一個。
        # 如果 seq_len == 0: 應該取誰？ Prompt 2 (Context) 還是 Prompt 1 (Learnable)?
        # 建議取最後一個 Prompt (Context)，因為它離歷史最近。
        
        # 計算索引:
        # Index 0: Learnable
        # Index 1: Context
        # Index 2: Item_0
        # ...
        # Last Item Index = (seq_len - 1) + 2 = seq_len + 1
        
        # 如果 seq_len == 0，我們希望取 Index 1 (Context)。
        # 公式: (0) + 1 = 1。
        # 所以通用公式是: target_index = seq_len + 1
        
        # (B, 1, 1)
        target_indices = (seq_lens + 1).view(B, 1, 1).expand(-1, 1, D)
        
        # (B, D)
        pooled_output = torch.gather(transformer_output, 1, target_indices).squeeze(1)
        
        return pooled_output

    def _build_feature_representations(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        覆寫以傳入 Learnable Prompt
        """
        # 1. 準備 Context (同父類)
        current_items_static = self.item_emb_w(batch['items'])
        batch_context = current_items_static.mean(dim=0)
        if self.training:
            self._update_context_ema(batch_context)
            context_to_use = batch_context
        else:
            context_to_use = self.global_context_ema
        context_batch = context_to_use.unsqueeze(0).expand(len(batch['users']), -1)
        
        # 2. 投影 Context Prompts
        prompt_context_A = self.context_proj_A(context_batch)
        prompt_context_B = self.context_proj_B(context_batch)
        
        # === User Tower ===
        static_u_emb = self.user_emb_w(batch['users'])
        
        hist_item_emb = self.item_emb_w(batch['item_history_matrix'])
        hist_cates = self.cates[batch['item_history_matrix']]
        hist_cates_emb = self.cate_emb_w(hist_cates)
        avg_cate_emb_for_hist = average_pooling(hist_cates_emb, self.cate_lens[batch['item_history_matrix']])
        hist_item_emb_with_cate = torch.cat([hist_item_emb, avg_cate_emb_for_hist], dim=2)

        # [呼叫]
        user_history_emb = self._run_transformer_with_prompt(
            hist_item_emb_with_cate, batch['item_history_len'], 
            prompt_context_A, self.learnable_prompt_A, # 傳入兩個 Prompt
            self.item_seq_transformer, self.item_seq_pos_emb
        )
        user_features = torch.cat([static_u_emb, user_history_emb.detach()], dim=-1)

        # === Item Tower ===
        static_item_emb = self.item_emb_w(batch['items'])
        item_cates = self.cates[batch['items']]
        item_cates_emb = self.cate_emb_w(item_cates)
        avg_cate_emb_for_item = average_pooling(item_cates_emb, self.cate_lens[batch['items']])
        item_emb_with_cate = torch.cat([static_item_emb, avg_cate_emb_for_item], dim=1)
        
        item_history_user_emb = self.user_emb_w(batch['user_history_matrix'])

        # [呼叫]
        item_history_emb = self._run_transformer_with_prompt(
            item_history_user_emb, batch['user_history_len'], 
            prompt_context_B, self.learnable_prompt_B, # 傳入兩個 Prompt
            self.user_seq_transformer, self.user_seq_pos_emb
        )
        item_features = torch.cat([item_emb_with_cate, item_history_emb.detach()], dim=-1)
        
        return user_features, item_features

    # inference 方法不需要重寫！
    # 因為我們繼承了 ContextPromptSASRec_MLP (父類)，
    # 但我們覆寫了 _build_feature_representations。
    # 父類的 inference 會呼叫我們覆寫後的 _build_feature_representations。
    # 唯一的問題是負樣本計算。
    # 父類的 inference 依賴快取 (item_history_buffer)。
    # 我們的 _build_feature_representations 已經計算了正確的 item_history_emb (雙 Prompt) 並存入快取。
    # 所以，直接使用父類的 inference 是安全的！它會直接讀取快取。