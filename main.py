import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import copy

# 引入 PyTorch 相關模組
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- 導入所有模型 ---
# (假設您的模型都已正確分離到這些檔案中)
from model_gcn import Hybrid_GNN_MLP , LightGCN_Only
from model_srgnn import PURE_SR_GNN, SR_GNN_MLP
from model_sasrec import PureSASRec, SASRec_MLP, CausalSASRec_MLP, ContextSASRec_MLP, DualPromptSASRec, HyperLoRASASRec
from model_mlp import EmbMLP  

from utils import (
    prepare_data_from_dfs,
    RecommendationDataset, 
    sample_negative_items, 
    build_lightgcn_graph,  
)
import random

# ######################################################################
# # 輔助函式：模型評估 (從主邏輯中提取)
# ######################################################################

def run_evaluation(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    item_history_dict: Dict[int, set], 
    seen_items_pool: set, 
    config: Dict[str, Any], 
    device: torch.device, 
    Ks: List[int], 
    sampling_size: int
) -> Dict[str, float]:
    """
    執行模型評估的輔助函式。
    
    Args:
        model: 待評估的模型。
        test_loader: 包含「正樣本」的測試資料載入器。
        item_history_dict: 用戶看過的物品歷史 (用於負採樣過濾)。
        seen_items_pool: 資料集中所有出現過的物品 (用於負採樣)。
        config: 全局設定檔。
        device: PyTorch 設備。
        Ks: Top-K 列表 (例如 [5, 10, 20])。
        sampling_size: 負採樣數量 (例如 99)。

    Returns:
        metrics (Dict[str, float]): 包含 GAUC, AUC, Recall@K, NDCG@K 的字典。
    """
    
    # 設置模型為評估模式
    model.eval() 
    
    total_recalls = np.zeros(len(Ks))
    total_ndcgs = np.zeros(len(Ks))
    all_per_sample_aucs = []
    all_user_ids_for_gauc = []
    total_positive_items = 0

    # 確保在評估時不計算梯度
    with torch.no_grad():
        # 僅遍歷正樣本
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch_cpu = {k: v for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = len(batch['users'])
            total_positive_items += batch_size

            # 1. 執行 1 vs M 負採樣 
            neg_item_ids_list = []
            for i in range(batch_size):
                item_j_id = batch['items'][i].item()
                user_i_id = batch_cpu['users'][i].item()
                seen_items = item_history_dict.get(user_i_id, set())
                
                neg_items = sample_negative_items(
                    item_pool=seen_items_pool,
                    seen_items_set=seen_items,
                    positive_item_id=item_j_id,
                    num_samples=sampling_size,
                    device=device
                )
                neg_item_ids_list.append(neg_items)
            
            # (B, M)
            neg_item_ids_batch = torch.stack(neg_item_ids_list) 

            # 2. 獲取模型推論 logits
            # pos_logits: (B,), neg_logits: (B, M)
            pos_logits, neg_logits, _ = model.inference(batch, neg_item_ids_batch)

            # 3. 組合 logits 以便排名
            # (B, 1)
            pos_logits = pos_logits.unsqueeze(1) 
            # (B, 1+M)
            all_logits = torch.cat([pos_logits, neg_logits], dim=1) 
            
            # 4. 計算排名 (Rank)
            # (all_logits > pos_logits) 會比較每個負樣本是否大於正樣本
            # .sum(dim=1) 計算有多少個負樣本 > 正樣本
            # 排名 0 = 最佳
            ranks = (all_logits > pos_logits).sum(dim=1).cpu().numpy()

            # 5. 計算 AUC (逐樣本)
            # (pos_logits > neg_logits) -> (B, M)
            auc_per_sample = (pos_logits > neg_logits).float().mean(dim=1)
            all_per_sample_aucs.extend(auc_per_sample.cpu().numpy())
            all_user_ids_for_gauc.extend(batch_cpu['users_raw'].numpy())

            # 6. 累加 Recall / NDCG
            for rank in ranks:
                for j, k in enumerate(Ks):
                    if rank < k:
                        total_recalls[j] += 1
                        total_ndcgs[j] += 1 / np.log2(rank + 2)
    
    # --- 彙總計算最終指標 ---
    metrics = {}
    if total_positive_items > 0:
        gauc_df = pd.DataFrame({'user': all_user_ids_for_gauc, 'auc': all_per_sample_aucs})
        metrics['auc'] = np.mean(all_per_sample_aucs)
        
        # 計算 GAUC (Group AUC, 按用戶分組的 AUC 均值，按樣本數加權)
        user_auc_mean = gauc_df.groupby('user')['auc'].mean()
        user_counts = gauc_df.groupby('user').size()
        metrics['gauc'] = (user_auc_mean * user_counts).sum() / user_counts.sum()
        
        # 計算 Recall / NDCG
        final_recalls = total_recalls / total_positive_items
        final_ndcgs = total_ndcgs / total_positive_items
        for k, rec, ndcg in zip(Ks, final_recalls, final_ndcgs):
            metrics[f'recall@{k}'] = rec
            metrics[f'ndcg@{k}'] = ndcg
            
    return metrics


# ######################################################################
# # 主實驗函式
# ######################################################################

def run_experiment(config: Dict[str, Any]):
    """
    主實驗執行函式 (串流推薦 v2.0，包含經驗回放與回溯評估)。
    """
    
    # --- 0. 設備設定 ---
    print("--- 0. Setting up device ---")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--- Using CUDA. ---")
    elif torch.backends.mps.is_available():
        # device = torch.device("mps")
        pass
    else:
        print("--- Using CPU. ---")

    random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))

    # --- 1. 數據 I/O ---
    print("--- 1. Loading raw data files ---")
    start_time = time.time()
    
    full_data_df = pd.read_parquet(config['data_path'])
    # (可選) Debug 模式，大幅減少資料量
    if config.get('debug_sample', 0) > 0:
        sample_ratio = config['debug_sample']
        print(f"--- [DEBUG] Sampling 1/{sample_ratio} of data ---")
        full_data_df = full_data_df.iloc[::sample_ratio]  
    full_meta_df = pd.read_parquet(config['meta_path'])

    # --- 2. 數據轉換 ---
    print("--- 2. Processing and remapping data ---")
    remapped_full_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map = prepare_data_from_dfs(
        full_data_df, full_meta_df, config
    )
    # 將模型超參和資料超參 (如 num_users) 合併
    hyperparams = {**config['model'], **hyperparams_updates}
    config['model'] = hyperparams # 更新 config，供模型初始化使用
    num_users = hyperparams['num_users']
    num_items = hyperparams['num_items']
    print(f"Global data processing finished. (Users: {num_users}, Items: {num_items})")
    
    # 轉換為 NumPy，用於模型初始化
    cates_np = np.array(cates)
    cate_lens_np = np.array(cate_lens)
    
    # --- [驗證] remapped_full_df (確保資料載入正確) ---
    if remapped_full_df.empty:
        print("!!! FATAL ERROR: remapped_full_df is empty! Check data paths or processing logic. !!!")
        return
    print("\n" + "="*30)
    print("--- Verifying remapped_full_df ---")
    print(f"Total rows in remapped_full_df: {len(remapped_full_df)}")
    print("Value counts for 'period' (Top 20):")
    print(remapped_full_df['period'].value_counts().sort_index().head(20))
    print(f"Config training range: {config['train_start_period']} to {config['num_periods'] - 1}")
    print("="*30 + "\n")

    # --- 準備評估所需的全局資訊 ---
    sampling_size = config['evaluation'].get('sampling_size', 99)
    Ks = config['evaluation'].get('Ks', [5, 10, 20, 50])
    print(f"--- Evaluation setup: 1 positive vs {sampling_size} negative samples. Ks = {Ks} ---")

    # --- 3. 進入學習率調參迴圈 ---
    for lr in config['learning_rates']:
        print(f"\n{'='*50}\nStarting run with Learning Rate: {lr}\n{'='*50}")
        dir_name_with_lr = f"{config['dir_name']}_lr{lr}"
        
        # --- [!!! NEW: 持續學習 (CL) 相關設定 !!!] ---
        # 1. 經驗回放 (ER) 緩衝區
        replay_buffer = pd.DataFrame()
        cl_config = config.get('continual_learning', {})
        replay_enabled = cl_config.get('replay_enabled', False)
        replay_buffer_max_size = cl_config.get('replay_buffer_size', 100000) # 緩衝區最大容量
        replay_add_size = cl_config.get('replay_add_size_per_period', 1000) # 每期結束後存入多少
        replay_sample_size = cl_config.get('replay_sample_size_per_period', 1000) # 每期訓練時取出多少
        if replay_enabled:
            print(f"--- [CL] Experience Replay enabled. Buffer size: {replay_buffer_max_size}, Sample size: {replay_sample_size} ---")

        # [!!! NEW: 讀取蒸餾設定 !!!]
        distill_config = config.get('distillation', {})
        distill_enabled = distill_config.get('enabled', False)
        kd_weight = distill_config.get('weight', 0.1)
        teacher_model = None # 初始化 Teacher

        if distill_enabled:
            print(f"--- [Distill] Self-Distillation Enabled. Weight (lambda): {kd_weight} ---")

        # 2. 回溯評估 (BWT) 儲存區
        # 我們儲存「測試集」的 DataLoader，以便未來回溯評估
        past_test_loaders = {} # Key: period_id, Value: DataLoader
        
        # 3. 評估指標儲存區
        results_over_periods = [] # 儲存「前向傳輸 (Forward Transfer)」的指標
        
        # 4. GNN 用的累加圖 (可選，目前使用瞬時圖)
        cumulative_interaction_df = pd.DataFrame() # 用於 GNN (累加圖模式)
        
        # 5. 全局評估用的 history (用於負採樣)
        item_history_dict = {} # K: user_id, V: set(item_ids)
        seen_items_pool = set()
        
        # --- 4. 進入增量訓練主迴圈 ---
        for period_id in range(config['train_start_period'], config['num_periods']):
            print(f"\n{'='*25} Period {period_id} {'='*25}")
            
            # [!!! NEW: 在訓練新 Period 之前，將當前模型備份為 Teacher !!!]
            # 只有從第 2 個訓練週期開始才有 Teacher (Period > start)
            if distill_enabled and period_id > config['train_start_period']:
                print("--- [Distill] Updating Teacher Model from previous period... ---")
                # 深拷貝當前模型 (此時 model 還是上個 period 的狀態)
                teacher_model = copy.deepcopy(model)
                teacher_model.eval() # Teacher 永遠是 eval 模式
                for param in teacher_model.parameters():
                    param.requires_grad = False # 凍結 Teacher
                teacher_model = teacher_model.to(device)
                print("--- [Distill] Teacher Model updated and frozen. ---")
                
            # --- 4.1 資料準備 ---
            
            # 1. 獲取「當前時期」的資料
            current_period_df = remapped_full_df[remapped_full_df['period'] == period_id]
            
            if current_period_df.empty:
                print(f"No training data for period {period_id}. Skipping.")
                continue
            
            # [驗證] 檢查正樣本
            positive_samples_count = (current_period_df['label'] == 1).sum()
            print(f"--- [Data] Period {period_id} loaded. Total rows: {len(current_period_df)}, Positive samples: {positive_samples_count}")
            if positive_samples_count == 0:
                print(f"--- [Warning] Period {period_id} has no positive samples. Skipping training.")
                # 即使沒有正樣本，我們也需要更新 history，但不進行訓練
                seen_items_pool.update(current_period_df['itemId'].unique())
                continue
                
            # 2. 獲取「測試集」(即 T+1 時期的資料)
            test_set = None
            test_loader_current = None
            if period_id >= config['test_start_period'] and (period_id + 1) < config['num_periods']:
                test_set = remapped_full_df[remapped_full_df['period'] == (period_id + 1)].iloc[::10] # (Debug: 可改為不採樣)
                
                if test_set is not None and not test_set.empty:
                    # 預先建立 DataLoader 並儲存，供 BWT 使用
                    test_set_pos = test_set[test_set['label'] == 1].copy()
                    if not test_set_pos.empty:
                        test_dataset_pos = RecommendationDataset(test_set_pos, {})
                        test_loader_current = DataLoader(
                            test_dataset_pos,
                            batch_size=config['model']['batch_size'],
                            shuffle=False,
                            num_workers=config.get('num_workers', 0)
                        )
                        # 儲存 T+1 的 DataLoader
                        past_test_loaders[period_id + 1] = test_loader_current
                        print(f"--- [Eval] Test set for T+1 (Period {period_id + 1}) prepared. Positive samples: {len(test_set_pos)}")


            # 3. GNN 建圖 (僅 GNN-based 模型需要)
            model_type = config.get('model_type', 'mlp')
            adj_matrix = None # 預設為 None
            
            if model_type in ['hybrid', 'lightgcn_only']:
                # GNN 建圖策略 (瞬時圖 vs 累加圖)
                graph_build_strategy = config.get('graph_build_strategy', 'instantaneous') # 'instantaneous' 或 'cumulative'
                
                current_interactions = current_period_df[current_period_df['label'] == 1].drop_duplicates(subset=['userId', 'itemId'])

                if graph_build_strategy == 'cumulative':
                    # 累加模式：使用 0~T 期的所有邊
                    cumulative_interaction_df = pd.concat([cumulative_interaction_df, current_interactions]).drop_duplicates(subset=['userId', 'itemId'])
                    interaction_df_for_graph = cumulative_interaction_df
                else:
                    # 瞬時模式 (預設)：只使用 T 期的邊
                    interaction_df_for_graph = current_interactions

                adj_matrix = build_lightgcn_graph(
                    interaction_df_for_graph,
                    num_users,
                    num_items,
                    device
                )
                print(f"--- [GNN] Graph built for Period {period_id}. Strategy: {graph_build_strategy}. Edges: {len(interaction_df_for_graph)} ---")
            
            # elif model_type in ['sr_gnn_mlp', 'sasrec_mlp', 'mlp']:
            #     # GNN 建圖策略 (從 Seq 建立圖)
            #     pass

            # 4. 更新評估用的 history (使用 T 期的所有資料)
            current_period_interactions_dict = current_period_df.groupby('userId')['itemId'].apply(set).to_dict()
            for user_id, item_set in current_period_interactions_dict.items():
                item_history_dict.setdefault(user_id, set()).update(item_set)
            seen_items_pool.update(current_period_df['itemId'].unique())
                
            
            # --- 4.2 訓練/驗證集切分 ---
            val_split_ratio = config.get('validation_split', 0.2)
            
            if val_split_ratio > 0:
                split_point = int(len(current_period_df) * (1.0 - val_split_ratio))
                if split_point == 0 or split_point == len(current_period_df):
                    val_set_df = None
                    train_set_current_df = current_period_df # 使用全部
                else:
                    train_set_current_df = current_period_df.iloc[:split_point]
                    val_set_df = current_period_df.iloc[split_point:]
            else:
                train_set_current_df = current_period_df
                val_set_df = None
            
            # --- [!!! NEW: 經驗回放 (ER) !!!] ---
            if replay_enabled and not replay_buffer.empty:
                # 1. 從緩衝區採樣
                num_replay_samples = min(replay_sample_size, len(replay_buffer))
                replay_samples_df = replay_buffer.sample(n=num_replay_samples)
                
                # 2. 將「當期訓練集」和「回放樣本」合併
                train_set_final_df = pd.concat([train_set_current_df, replay_samples_df])
                
                print(f"--- [CL] Training with Experience Replay: {len(train_set_current_df)} (current) + {len(replay_samples_df)} (replay) = {len(train_set_final_df)} samples ---")
            else:
                train_set_final_df = train_set_current_df
            # --- [ ER 結束 ] ---

            # 3. 建立 DataLoaders 
            train_dataset = RecommendationDataset(train_set_final_df, {}) 
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['model']['batch_size'], 
                shuffle=True, # 訓練集開啟 shuffle
                num_workers=config.get('num_workers', 0)
            )
            
            val_loader = None
            if val_set_df is not None and not val_set_df.empty:
                val_dataset = RecommendationDataset(val_set_df, {})
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config['model']['batch_size'],
                    shuffle=False, 
                    num_workers=config.get('num_workers', 0)
                )

            # --- 4.3 模型初始化與載入 ---
            print(f"--- Building model, type: {model_type} ---")
            
            # (根據 model_type 初始化對應的模型)
            if model_type == 'hybrid':
                model = Hybrid_GNN_MLP(num_users, num_items, hyperparams, adj_matrix, cates_np, cate_lens_np).to(device)
            elif model_type == 'lightgcn_only':
                model = LightGCN_Only(num_users, num_items, hyperparams, adj_matrix).to(device)
            elif model_type == 'sr_gnn':
                model = PURE_SR_GNN(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'sr_gnn_mlp':
                model = SR_GNN_MLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'pure_sasrec':
                model = PureSASRec(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'sasrec_mlp':
                model = SASRec_MLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'mlp':
                model = EmbMLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'causal_sasrec_mlp':
                model = CausalSASRec_MLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'context_sasrec_mlp':
                model = ContextSASRec_MLP(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'dual_prompt_sasrec':
                model = DualPromptSASRec(cates_np, cate_lens_np, hyperparams, config).to(device)
            elif model_type == 'hyper_lora_sasrec':
                model = HyperLoRASASRec(cates_np, cate_lens_np, hyperparams, config).to(device)
            else:
                raise ValueError(f"未知的 model_type: {model_type}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.0))
            
            # [!!! 關鍵: 載入上一個 Period 的權重 !!!]
            if period_id > config['train_start_period']:
                prev_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id-1}')
                ckpt_path = os.path.join(prev_ckpt_dir, 'best_model.pth')
                if os.path.exists(ckpt_path):
                    try:
                        print(f"--- [CL] Loading weights from: {ckpt_path} ---")
                        state_dict = torch.load(ckpt_path, map_location=device) 
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"--- [Warning] Could not load weights: {e} ---")
                else:
                    print(f"--- [Warning] No checkpoint found at {ckpt_path}. Training from scratch for this period. ---")

            # --- 5. 進入訓練迴圈 (Epochs) ---
            max_epochs = config.get('max_epochs', 10)
            patience = config.get('patience', 3)
            patience_counter = 0
            best_val_loss = float('inf')
            
            period_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id}')
            os.makedirs(period_ckpt_dir, exist_ok=True)
            best_model_path = os.path.join(period_ckpt_dir, 'best_model.pth')
            
            pbar_epochs = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=True)
            for epoch_id in pbar_epochs:
                
                # --- (a) 訓練階段 ---
                model.train()
                losses = []
                for batch in train_loader:
                    
                    
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    loss = model.calculate_loss(batch)
                    # loss = model.calculate_loss(
                    #     batch, 
                    #     teacher_model=teacher_model if distill_enabled else None, 
                    #     kd_weight=kd_weight
                    # )
                    
                    # 檢查 NaN
                    if torch.isnan(loss):
                        print(f"--- [FATAL] NaN loss detected in Epoch {epoch_id}! Skipping batch. ---")
                        continue
                        
                    loss.backward() 
                    optimizer.step()
                    losses.append(loss.item())

                # 檢查訓練是否有效
                if not losses:
                    print(f"--- [Warning] Epoch {epoch_id}: No valid training batches. Skipping epoch. ---")
                    avg_train_loss = 0.0
                    # 如果所有批次都被跳過，可能無法進行早停
                    if epoch_id == 1:
                        print("--- [Warning] All training batches were skipped. Stopping training for this period. ---")
                        break # 提前終止此 Period 的訓練
                else:
                    avg_train_loss = np.mean(losses)
                
                # --- (b) 驗證階段 ---
                if val_loader is None:
                    # 如果沒有驗證集，則始終儲存模型
                    torch.save(model.state_dict(), best_model_path)
                    continue

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                            
                        batch = {k: v.to(device) for k, v in batch.items()}
                        loss = model.calculate_loss(batch)
                        if not torch.isnan(loss):
                            val_losses.append(loss.item())
                        
                # 處理 val_losses 為空的情況
                if not val_losses:
                    print(f"--- [Warning] Epoch {epoch_id}: No valid validation batches. ---")
                    avg_val_loss = float('inf') # 設置為無窮大，以便早停 (如果需要)
                else:
                    avg_val_loss = np.mean(val_losses)

                # --- (c) 決策階段 (Early Stopping) ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break 

                pbar_epochs.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", patience=f" {patience_counter}/{patience}")
            
            print(f"--- Training finished for period {period_id}. Best model saved to {best_model_path} ---")

            # --- 6. 評估階段 (Forward & Backward Transfer) ---

            # (檢查 T+1 的測試集是否存在)
            if test_loader_current is not None:
                
                # 載入本期訓練好的最佳模型
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                else:
                    print(f"--- WARNING: No best model found. Evaluating with last-epoch model. ---")
                
                # (GNN 模型需要手動更新 buffer)
                if model_type == 'hybrid' or model_type == 'lightgcn_only':
                    with torch.no_grad():
                        model.update_gnn_buffer()
                
                # --- 6.1 [!!! NEW: 回溯評估 (BWT) !!!] ---
                # 評估 Model(T) 在所有過去 TestSet (T, T-1, ...) 上的表現
                print(f"--- [CL] Running Backward Transfer (BWT) Evaluation for Model({period_id}) ---")
                backward_metrics_list = []
                # 遍歷所有已儲存的「過去的」測試集
                for past_period_id, past_loader in past_test_loaders.items():
                    # (跳過 T+1 的測試集，那是 FWT)
                    if past_period_id == (period_id + 1):
                        continue
                        
                    print(f"    - Evaluating Model({period_id}) on TestSet(Period {past_period_id})...")
                    past_metrics = run_evaluation(
                        model, past_loader, item_history_dict, seen_items_pool, 
                        config, device, Ks, sampling_size
                    )
                    backward_metrics_list.append(past_metrics)
                    print(f"    - TestSet(Period {past_period_id}) GAUC: {past_metrics.get('gauc', 0.0):.4f}")
                
                if backward_metrics_list:
                    # 計算平均回溯效能
                    avg_bwt_gauc = np.mean([m.get('gauc', 0.0) for m in backward_metrics_list])
                    print(f"--- [CL] Average BWT GAUC for Model({period_id}): {avg_bwt_gauc:.4f} ---")


                # --- 6.2 [!!! (原) 前向評估 (FWT) !!!] ---
                # 評估 Model(T) 在 TestSet(T+1) 上的表現
                print(f"--- [CL] Running Forward Transfer (FWT) Evaluation for Model({period_id}) on TestSet(Period {period_id + 1}) ---")
                
                metrics_forward = run_evaluation(
                    model, test_loader_current, item_history_dict, seen_items_pool, 
                    config, device, Ks, sampling_size
                )
                
                # 儲存 T+1 的結果，作為本期的主要指標
                results_over_periods.append(metrics_forward)
                
                print(f"\n--- Period {period_id} (Model) -> Period {period_id + 1} (Test) Evaluation Finished ---")
                print(f"  - GAUC     : {metrics_forward.get('gauc', 0.0):.4f}")
                print(f"  - AUC      : {metrics_forward.get('auc', 0.0):.4f}")
                print("  -----------------------------------------------------")
                if metrics_forward:
                    for k in Ks:
                        print(f"  - Recall@{k:<2} : {metrics_forward.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {metrics_forward.get(f'ndcg@{k}', 0.0):.4f}")
                else:
                    print("  - No positive samples for Recall/NDCG in this period.")
                print("  -----------------------------------------------------")

            # --- [!!! NEW: 經驗回放 (ER) 更新緩衝區 !!!] ---
            if replay_enabled:
                # 1. 從「當期」資料中採樣，準備存入
                num_to_add = min(replay_add_size, len(current_period_df))
                samples_to_add = current_period_df.sample(n=num_to_add)
                
                # 2. 合併
                replay_buffer = pd.concat([replay_buffer, samples_to_add])
                
                # 3. 如果超過容量，則進行「蓄水池採樣」(隨機丟棄舊資料)
                if len(replay_buffer) > replay_buffer_max_size:
                    replay_buffer = replay_buffer.sample(n=replay_buffer_max_size)
                    
                print(f"--- [CL] Replay buffer updated. New size: {len(replay_buffer)} / {replay_buffer_max_size} ---")
            
        # --- 7. 學習率總結報告 ---
        if results_over_periods:
            print(f"\n{'='*20} [ 學習率 {lr} 總結報告 (FWT) ] {'='*20}")
            avg_metrics = {}
            report_metric_keys = ['gauc', 'auc'] + [f'recall@{k}' for k in Ks] + [f'ndcg@{k}' for k in Ks]
            
            for key in report_metric_keys:
                valid_values = [m.get(key) for m in results_over_periods if m.get(key) is not None and np.isfinite(m.get(key))]
                avg_metrics[key] = np.mean(valid_values) if valid_values else 0.0
                
            print("\n--- Average Forward Transfer Metrics ---")
            print(f"  - GAUC     : {avg_metrics.get('gauc', 0.0):.4f}")
            print(f"  - AUC      : {avg_metrics.get('auc', 0.0):.4f}")
            print("-------------------------------------------------------")
            for k in Ks:
                print(f"  - Recall@{k:<2} : {avg_metrics.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {avg_metrics.get(f'ndcg@{k}', 0.0):.4f}")
            print("-------------------------------------------------------")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Recommendation Experiment Framework (v2.0)")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    
    # --- 執行實驗 ---
    run_experiment(config)