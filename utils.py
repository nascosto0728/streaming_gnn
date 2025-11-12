# utils.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import random 

import torch
from torch.utils.data import Dataset

def pad_sequences_native(
    sequences: List[list], 
    maxlen: int, 
    dtype: str = 'int32', 
    padding: str = 'post', 
    truncating: str = 'post', 
    value: int = 0
) -> np.ndarray:
    """
    用純 NumPy 實現的 Keras pad_sequences 的替代品。
    """
    truncated_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                truncated_sequences.append(seq[-maxlen:])
            elif truncating == 'post':
                truncated_sequences.append(seq[:maxlen])
            else:
                raise ValueError(f"Truncating type '{truncating}' not understood")
        else:
            truncated_sequences.append(seq)
            
    num_samples = len(truncated_sequences)
    padded_matrix = np.full((num_samples, maxlen), value, dtype=dtype)

    for i, seq in enumerate(truncated_sequences):
        seq_len = len(seq)
        if padding == 'pre':
            padded_matrix[i, -seq_len:] = seq
        elif padding == 'post':
            padded_matrix[i, :seq_len] = seq
        else:
            raise ValueError(f"Padding type '{padding}' not understood")
            
    return padded_matrix


class RecommendationDataset(Dataset):
    """
    PyTorch Dataset Class for loading recommendation data.
    """
    def __init__(self, data_df: pd.DataFrame, his_data: Dict):
        self.data_df = data_df.reset_index(drop=True)
        self.his = his_data

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_row = self.data_df.iloc[idx]

        item_history_seq = pad_sequences_native([sample_row['itemSeq']], maxlen=30, padding='post', truncating='post', dtype='int32')[0]
        user_history_seq = pad_sequences_native([sample_row['userSeq']], maxlen=30, padding='post', truncating='post', dtype='int32')[0]
        
        
        batch_data = {
            'users': sample_row['userId'].astype(np.int32),
            'items': sample_row['itemId'].astype(np.int32),
            'labels': sample_row['label'].astype(np.int32),
            'users_raw': sample_row['userId_raw'].astype(np.int64),
            'items_raw': sample_row['itemId_raw'].astype(np.int64),
            'item_history_matrix': item_history_seq,
            'item_history_len': np.minimum(len(sample_row['itemSeq']), 30).astype(np.int32),
            'user_history_matrix': user_history_seq,
            'user_history_len': np.minimum(len(sample_row['userSeq']), 30).astype(np.int32),

            'period': sample_row['period'].astype(np.int64),
        }
        
        return batch_data


def process_cate(cate_ls: list) -> Tuple[np.ndarray, list]:
    cate_lens = [len(cate) for cate in cate_ls]
    max_len = max(cate_lens) if cate_lens else 0
    cate_seqs_matrix = np.zeros([len(cate_ls), max_len], np.int32)
    for i, cate_seq in enumerate(cate_ls):
        cate_seqs_matrix[i, :len(cate_seq)] = cate_seq
    return cate_seqs_matrix, cate_lens

def prepare_data_from_dfs(full_data_df: pd.DataFrame, full_meta_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, list, Dict, Dict, Dict]:
    print("Backing up raw user and item IDs...")
    full_data_df['userId_raw'] = full_data_df['userId']
    full_data_df['itemId_raw'] = full_data_df['itemId']

    print("Cleaning sequence string format...")
    for col in ['itemSeq', 'userSeq']:
        full_data_df[col] = full_data_df[col].fillna('').apply(lambda x: [int(i) for i in x.split('#') if i])
    full_meta_df['cateId'] = full_meta_df['cateId'].fillna('').apply(lambda x: [c for c in x.split('#') if c])

    print("Building full category map...")
    all_cate_names = set()
    _ = [all_cate_names.update(seq) for seq in full_meta_df['cateId']]
    full_cate_map = {raw_name: i for i, raw_name in enumerate(sorted(list(all_cate_names)))}
    config['model']['num_cates'] = len(full_cate_map)

    print("--- Starting static global data processing ---")
    all_user_ids = set(full_data_df['userId_raw'].unique())
    _ = [all_user_ids.update(seq) for seq in full_data_df['userSeq']]
    all_item_ids = set(full_data_df['itemId_raw'].unique())
    _ = [all_item_ids.update(seq) for seq in full_data_df['itemSeq']]
    user_map = {raw_id: i for i, raw_id in enumerate(sorted(list(all_user_ids)))}
    item_map = {raw_id: i for i, raw_id in enumerate(sorted(list(all_item_ids)))}
    print(f"Global maps created. Total users: {len(user_map)}, Total items: {len(item_map)}")

    def map_seq(seq, id_map):
        return [id_map.get(i) for i in seq if i in id_map]
    remapped_df = full_data_df.copy()
    remapped_df['userId'] = remapped_df['userId_raw'].map(user_map)
    remapped_df['itemId'] = remapped_df['itemId_raw'].map(item_map)
    remapped_df['userSeq'] = remapped_df['userSeq'].apply(lambda seq: map_seq(seq, user_map))
    remapped_df['itemSeq'] = remapped_df['itemSeq'].apply(lambda seq: map_seq(seq, item_map))
    remapped_df.dropna(subset=['userId', 'itemId'], inplace=True)
    remapped_df = remapped_df.astype({'userId': 'int32', 'itemId': 'int32'})
    print("Full DataFrame remapped.")

    meta_df = full_meta_df.copy()
    meta_df['itemId'] = meta_df['itemId'].map(item_map)
    meta_df.dropna(subset=['itemId'], inplace=True)
    meta_df['cateId'] = meta_df['cateId'].apply(lambda seq: map_seq(seq, full_cate_map))
    item_cate_map = pd.Series(meta_df['cateId'].values, index=meta_df['itemId']).to_dict()
    cate_ls = [item_cate_map.get(i, []) for i in range(len(item_map))]
    cates, cate_lens = process_cate(cate_ls)

    hyperparams_updates = {'num_users': len(user_map), 'num_items': len(item_map)}
    print("Static metadata (cates, hyperparams) created.")
    
    return remapped_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map


def sample_negative_users(
    user_pool: set,
    seen_users_set: set,
    positive_user_id: int,
    num_samples: int,
    device: torch.device
) -> torch.Tensor:
    """
    Args:
        seen_users_set (set): 已經與目標物品互動過的用戶 ID 集合。
        positive_user_id (int): 當前正樣本的用戶 ID。
        num_samples (int): 需要抽取的負樣本數量 (M)。
        device (torch.device): 目標設備 (用於創建最終的 Tensor)。

    Returns:
        torch.Tensor: 包含 num_samples 個負樣本用戶 ID 的 LongTensor。
    """

    random.seed(42)
    
    # *** 核心修改：從 user_pool 而不是 range(num_users) 開始 ***
    candidate_negatives = user_pool - seen_users_set - {positive_user_id}
    n_candidates = len(candidate_negatives)

    if n_candidates == 0:
        # 極端情況：所有用戶都互動過或就是正樣本，隨機選 M 個非正樣本用戶
        candidate_negatives = set(range(user_pool)) - {positive_user_id}
        if not candidate_negatives: # 如果只有一個用戶，只能重複選自己（理論上不應發生）
             return torch.tensor([positive_user_id] * num_samples, dtype=torch.long, device=device)
        sampled_ids = random.choices(list(candidate_negatives), k=num_samples)

    elif n_candidates < num_samples:
        # 候選不足，允許重複採樣 (sample with replacement)
        sampled_ids = random.choices(list(candidate_negatives), k=num_samples)
    else:
        # 候選充足，不重複採樣 (sample without replacement)
        sampled_ids = random.sample(list(candidate_negatives), k=num_samples)

    return torch.tensor(sampled_ids, dtype=torch.long, device=device)



# ===================================================================
# GNN (LightGCN) 專用工具 (*** 以下為新增 ***)
# ===================================================================
from torch.utils.data import Dataset
# 我們需要 PyG 的稀疏工具
try:
    from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
    from torch_geometric.utils.num_nodes import maybe_num_nodes
except ImportError:
    print("警告：未找到 PyTorch Geometric (torch_geometric)。")
    print("請執行 `pip install torch_geometric torch_sparse`")
import scipy.sparse as sp

def build_lightgcn_graph(interaction_df: pd.DataFrame, num_users: int, num_items: int, device: torch.device) -> torch.sparse.FloatTensor:
    """
    建立 LightGCN 所需的「對稱正規化鄰接矩陣」。
    D^(-1/2) * A * D^(-1/2)
    
    Args:
        interaction_df (pd.DataFrame): 包含 'userId' 和 'itemId' 的 DataFrame (必須是內部 ID)
        num_users (int): 用戶總數
        num_items (int): 物品總數
        device (torch.device): 目標設備
    
    Returns:
        torch.sparse.FloatTensor: 正規化後的鄰接矩陣
    """
    print(f"--- Building LightGCN Graph (N={num_users + num_items}) ---")
    N = num_users + num_items
    
    # 1. 獲取邊 (edge)
    user_ids = interaction_df['userId'].values
    item_ids = interaction_df['itemId'].values
    
    # 2. 偏移 Item ID (GNN 的標準作法)
    # 節點 0~N_u-1 是 users
    # 節點 N_u~N-1 是 items
    item_ids_global = item_ids + num_users
    
    # 3. 建立 PyG 格式的 edge_index
    # 先用 numpy.stack 把它們堆疊成一個 (2, E) 的 ndarray
    edges_part1_np = np.stack([user_ids, item_ids_global])# (u, i) 互動
    edges_part2_np = np.stack([item_ids_global, user_ids])# (i, u) 互動 (因為圖是無向的)
    
    # 3. 建立 PyG 格式的 edge_index
    edges_part1 = torch.from_numpy(edges_part1_np).to(dtype=torch.long)
    edges_part2 = torch.from_numpy(edges_part2_np).to(dtype=torch.long)
    
    edge_index = torch.cat([edges_part1, edges_part2], dim=1)
    
    # 4. 建立稀疏鄰接矩陣 'A' (A = A + I, 包含自環)
    # PyG 的 from_scipy_sparse_matrix 會自動處理自環和正規化
    
    # 建立一個 (N, N) 的單位矩陣 (I)
    eye_edge_index = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    
    # A' = A + I
    edge_index = torch.cat([edge_index, eye_edge_index], dim=1)
    
    # 獲取邊的權重 (全為 1)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    # 5. 計算 LightGCN 的對稱正規化: D^(-1/2) * A' * D^(-1/2)
    row, col = edge_index
    
    # 計算度 (Degree) D
    deg = torch.zeros(N, dtype=torch.float32)
    deg = deg.scatter_add_(0, row, edge_weight) # D = A'.sum(axis=1)
    
    # D^(-1/2)
    deg_inv_sqrt = deg.pow_(-0.5)
    # 處理 deg 為 0 的情況 (雖然 A+I 應該不會有)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    # 建立 D^(-1/2) * A' * D^(-1/2) 的值
    # 矩陣中 (i, j) 的值 = norm_i * A'_ij * norm_j
    normed_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    # 6. 建立最終的稀疏 Tensor
    adj_matrix_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=normed_edge_weight,
        size=torch.Size([N, N])
    )

    adj_matrix_sparse = adj_matrix_sparse.coalesce().to(device)
    
    print(f"--- Graph built. Shape: {adj_matrix_sparse.shape}, Device: {device} ---")
    return adj_matrix_sparse


class BPRDataset(Dataset):
    """
    LightGCN / GNN 專用的 BPR 訓練資料集。
    
    它只在 `__getitem__` 時即時採樣負樣本。
    """
    def __init__(self, interaction_df: pd.DataFrame, num_items: int):
        super().__init__()
        self.num_items = num_items
        
        # 1. 獲取所有用戶
        self.users = torch.tensor(interaction_df['userId'].unique(), dtype=torch.long)
        
        # 2. 建立「用戶 -> 正樣本物品集」的字典
        self.user_pos_item_map = interaction_df.groupby('userId')['itemId'].apply(set)
        
        # 3. 獲取所有物品 ID (用於負採樣)
        self.all_items = set(range(num_items))
        print(f"--- BPRDataset: {len(self.users)} users, {self.num_items} items prepared. ---")

    def __len__(self):
        # 我們每個 epoch 只訓練一次「有互動」的用戶
        return len(self.users)

    def __getitem__(self, index):
        user_id = self.users[index].item()
        
        # 1. 隨機選一個該用戶的正樣本
        pos_items = self.user_pos_item_map[user_id]
        pos_item_id = random.choice(list(pos_items))
        
        # 2. 隨機選一個負樣本
        neg_item_id = self.sample_negative(pos_items)
        
        return user_id, pos_item_id, neg_item_id

    def sample_negative(self, pos_items_set: set) -> int:
        """
        隨機採樣一個不在 pos_items_set 中的 item
        """
        while True:
            # 隨機選一個 item (0 ~ num_items-1)
            neg_item_id = random.randint(0, self.num_items - 1)
            if neg_item_id not in pos_items_set:
                return neg_item_id