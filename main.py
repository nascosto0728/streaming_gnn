import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm

# 引入 PyTorch 相關模組
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Hybrid_GNN_MLP , LightGCN_Only
from model_mlp import EmbMLP  
from utils import (
    prepare_data_from_dfs,
    RecommendationDataset, # (復活) 我們現在用這個
    sample_negative_items, # (保留) 為了 Test
    build_lightgcn_graph,  # (保留) 為了建圖
)
import random


def run_experiment(config: Dict[str, Any]):
    """
    主實驗執行函式 (*** Hybrid GNN-MLP 重構版 ***)。
    """
    # --- 0. 設備設定 ---
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        # device = torch.device("mps")
        pass
    else:
        print("--- Using CPU. ---")

    random.seed(42)
    # --- 1. 數據 I/O ---
    print("--- Loading raw data files ---")
    start_time = time.time()
    
    full_data_df = pd.read_parquet(config['data_path'])
    if config.get('debug_sample', True):
        full_data_df = full_data_df.iloc[::3]  
        full_data_df = full_data_df[full_data_df['period'] < 14]
    full_meta_df = pd.read_parquet(config['meta_path'])

    # --- 2. 數據轉換 ---
    # (MODIFIED) 我們的 Hybrid 模型需要 cates 和 cate_lens
    remapped_full_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map = prepare_data_from_dfs(
        full_data_df, full_meta_df, config
    )
    hyperparams = {**config['model'], **hyperparams_updates}
    config['model'] = hyperparams
    num_users = hyperparams['num_users']
    num_items = hyperparams['num_items']
    print(f"Global data processing finished. (Users: {num_users}, Items: {num_items})")
    
    # (NEW) 我們的 Hybrid 模型需要 cates_np
    cates_np = np.array(cates)
    cate_lens_np = np.array(cate_lens)
    
    # (GNN 預設使用 Xavier 初始化，我們刪除所有 SBERT/KGE 的 .npy 載入邏輯)

    # --- 準備評估所需的全局資訊 ---
    sampling_size = config['evaluation'].get('sampling_size', 99)
    Ks = [5, 10, 20, 50]
    print(f"--- Evaluation setup: 1 positive vs {sampling_size} negative samples ---")

    # --- 3. 進入學習率調參迴圈 ---
    for lr in config['learning_rates']:
        print(f"\n{'='*50}\nStarting run with Learning Rate: {lr}\n{'='*50}")
        dir_name_with_lr = f"{config['dir_name']}_lr{lr}"
        results_over_periods = []
        item_history_dict = {}
        seen_items_pool = set()
        
        # ( GNN 需要一個「不斷增長」的互動 DataFrame
        incremental_interaction_df = pd.DataFrame()

        # --- 4. 進入增量訓練主迴圈 ---
        for period_id in range(config['train_start_period'], config['num_periods']):
            print(f"\n{'='*25} Period {period_id} {'='*25}")
            
            # --- (*** 核心修改 3: Hybrid 資料準備 ***) ---
            
            # 1. 獲取「當前時期」的資料
            current_period_df = remapped_full_df[remapped_full_df['period'] == period_id]
            
            # 2. 獲取測試集 
            test_set = None
            if period_id >= config['test_start_period'] and (period_id + 1) < config['num_periods']:
                test_set = remapped_full_df[remapped_full_df['period'] == period_id + 1].iloc[::10]

            if current_period_df.empty:
                print(f"No training data for period {period_id}. Skipping.")
                continue
                
            # 3. 互動 (GNN 建圖用) 
            # incremental_interaction_df = current_period_df[current_period_df['label'] == 1].drop_duplicates(subset=['userId', 'itemId'])
            incremental_interaction_df = pd.concat([incremental_interaction_df, 
                                                    current_period_df[current_period_df['label'] == 1]]).drop_duplicates(subset=['userId', 'itemId'])
            
            
            # 4. 建立「增量圖」 
            adj_matrix = build_lightgcn_graph(
                incremental_interaction_df,
                num_users,
                num_items,
                device
            )
            # [*** 在此加入驗證 ***]
            num_nodes = num_users + num_items
            num_edges_in_period = incremental_interaction_df.shape[0]
            num_nnz_in_adj = adj_matrix._nnz() # 包含雙向 + 自環
            graph_density = num_nnz_in_adj / (num_nodes * num_nodes)
            
            print(f"--- [驗證] Period {period_id} 圖稀疏性 ---")
            print(f"    - 總節點數 (U+I): {num_nodes}")
            print(f"    - 當期互動邊數: {num_edges_in_period}")
            print(f"    - Adj 矩陣非零元 (含自環): {num_nnz_in_adj}")
            print(f"    - 圖密度 (Density): {graph_density:.8f}")
            # [*** 驗證結束 ***]

            # 5. 更新評估用的 history 
            current_period_interactions_dict = current_period_df.groupby('userId')['itemId'].apply(set).to_dict()
            for user_id, item_set in current_period_interactions_dict.items():
                item_history_dict.setdefault(user_id, set()).update(item_set)
            seen_items_pool.update(current_period_df['itemId'].unique())
                
            
            # 1. 獲取早停參數 
            val_split_ratio = config.get('validation_split', 0.2)
            
            # 2. 切分「當前時期」的「訓練集」和「驗證集」
            if val_split_ratio > 0:
                split_point = int(len(current_period_df) * (1.0 - val_split_ratio))
                if split_point == 0 or split_point == len(current_period_df):
                    val_set = None
                    train_set = current_period_df # 使用全部
                else:
                    train_set = current_period_df.iloc[:split_point]
                    val_set = current_period_df.iloc[split_point:]
            else:
                train_set = current_period_df
                val_set = None
            

            # 3. 建立 DataLoaders 
            train_dataset = RecommendationDataset(train_set, {}) 
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['model']['batch_size'], 
                shuffle=False, 
                num_workers=config.get('num_workers', 0)
            )
            
            val_loader = None
            if val_set is not None and not val_set.empty:
                val_dataset = RecommendationDataset(val_set, {})
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config['model']['batch_size'],
                    shuffle=False, 
                    num_workers=config.get('num_workers', 0)
                )

            model_type = config.get('model_type', 'mlp') # 預設為 'mlp'
            print(f"--- Building model, type: {model_type} ---")

            if model_type == 'hybrid':
                model = Hybrid_GNN_MLP(
                    num_users, 
                    num_items, 
                    hyperparams,
                    adj_matrix, # 傳入「當前」的圖
                    cates_np,
                    cate_lens_np
                ).to(device)

            elif model_type == 'lightgcn_only':
                model = LightGCN_Only(
                    num_users,
                    num_items,
                    hyperparams,
                    adj_matrix # 只需要傳入圖
                ).to(device)
            
            elif model_type == 'mlp':
                model = EmbMLP(
                    cates_np,
                    cate_lens_np,
                    hyperparams=hyperparams,
                    train_config=config,
                    item_init_vectors=None, # (傳入 SBERT/KGE)
                    cate_init_vectors=None  # (傳入 SBERT)
                ).to(device)
            
            else:
                raise ValueError(f"未知的 model_type: {model_type}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_model_path = '' 
            if period_id > config['train_start_period']:
                prev_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id-1}')
                ckpt_path = os.path.join(prev_ckpt_dir, 'best_model.pth') # <--- 換回 .pth
                if os.path.exists(ckpt_path):
                    try:
                        model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    except Exception as e:
                        print(f"Attempting to restore model weights from: {ckpt_path}")
                        print(f"Could not load embedding weights: {e}")

            # --- 5. 進入訓練迴圈 ---
            max_epochs = config.get('max_epochs', 100)
            patience = config.get('patience', 10)
            patience_counter = 0
            best_val_loss = float('inf')
            
            period_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id}')
            os.makedirs(period_ckpt_dir, exist_ok=True)
            # (MODIFIED) 我們存「整個模型」
            best_model_path = os.path.join(period_ckpt_dir, 'best_model.pth')
            
            pbar_epochs = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=True)
            for epoch_id in pbar_epochs:
                start_time_epoch = time.time()
                
                # --- (a) 訓練階段 ---
                model.train()
                

                losses = []
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    optimizer.zero_grad()
                    
                    
                    # 計算 MLP Loss (BCE)
                    loss = model.calculate_loss(batch)
                    
                    loss.backward() 
                    optimizer.step()
                    
                    losses.append(loss.item())

                avg_train_loss = np.mean(losses)
                
                # --- (b) 驗證階段 ---
                if val_loader is None:
                    torch.save(model.state_dict(), best_model_path)
                    continue

                model.eval()
                val_losses = []
                
                with torch.no_grad():
                
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        # 計算 MLP Loss (BCE)
                        loss = model.calculate_loss( batch)
                        val_losses.append(loss.item())
                        
                avg_val_loss = np.mean(val_losses)

                # --- (c) 決策階段 (Early Stopping) ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # (MODIFIED) *** 儲存「整個模型」 ***
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break 

                pbar_epochs.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", patience=f" {patience_counter}/{patience}")
            
            print(f"--- Training finished for period {period_id}. Best model saved to {best_model_path} ---")

            # --- 6. 評估階段 ---
            if test_set is not None and not test_set.empty:
                seen_items_pool.update(test_set['itemId'].unique())
                
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                else:
                    print(f"--- WARNING: No best model found. Evaluating with last-epoch model. ---")
                
                model.eval() 
                if model_type == 'hybrid' or model_type == 'lightgcn_only':
                    with torch.no_grad():
                        model.update_gnn_buffer()
                
                test_set_pos = test_set[test_set['label'] == 1].copy()
                total_positive_items = len(test_set_pos)
                
                total_recalls = np.zeros(len(Ks))
                total_ndcgs = np.zeros(len(Ks))
                all_per_sample_aucs = []
                all_user_ids_for_gauc = []

                if total_positive_items > 0:
                    test_dataset_pos = RecommendationDataset(test_set_pos, {})
                    test_loader_pos = DataLoader(
                        test_dataset_pos,
                        batch_size=config['model']['batch_size'],
                        shuffle=False,
                        num_workers=config.get('num_workers', 0)
                    )

                    with torch.no_grad():
                        for batch in tqdm(test_loader_pos, desc="Evaluating"):
                            batch_cpu = {k: v for k, v in batch.items()}
                            batch = {k: v.to(device) for k, v in batch.items()}
                            batch_size = len(batch['users'])

                            # 1. 負採樣 
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
                            neg_item_ids_batch = torch.stack(neg_item_ids_list) # (B, M)

                            # 2. (*** GNN-MLP 評估邏輯 ***)
                            pos_logits, neg_logits, _ = model.inference(batch, neg_item_ids_batch)

                            # 3. 組合 logits 
                            pos_logits = pos_logits.unsqueeze(1) 
                            all_logits = torch.cat([pos_logits, neg_logits], dim=1) 
                            
                            # 4. 排名 
                            ranks = (all_logits > pos_logits).sum(dim=1).cpu().numpy()

                            # 5. AUC 
                            auc_per_sample = (pos_logits > neg_logits).float().mean(dim=1)
                            all_per_sample_aucs.extend(auc_per_sample.cpu().numpy())
                            all_user_ids_for_gauc.extend(batch_cpu['users_raw'].numpy())

                            # 6. 累加 Recall/NDCG 
                            for rank in ranks:
                                for j, k in enumerate(Ks):
                                    if rank < k:
                                        total_recalls[j] += 1
                                        total_ndcgs[j] += 1 / np.log2(rank + 2)
                
                metrics = {}
                if total_positive_items > 0:
                    gauc_df = pd.DataFrame({'user': all_user_ids_for_gauc, 'auc': all_per_sample_aucs})
                    metrics['auc'] = np.mean(all_per_sample_aucs)
                    user_auc_mean = gauc_df.groupby('user')['auc'].mean()
                    user_counts = gauc_df.groupby('user').size()
                    metrics['gauc'] = (user_auc_mean * user_counts).sum() / user_counts.sum()
                    final_recalls = total_recalls / total_positive_items
                    final_ndcgs = total_ndcgs / total_positive_items
                    for k, rec, ndcg in zip(Ks, final_recalls, final_ndcgs):
                        metrics[f'recall@{k}'] = rec
                        metrics[f'ndcg@{k}'] = ndcg
                results_over_periods.append(metrics)
                
                print(f"\n--- Period {period_id} Evaluation Finished ---")
                print(f"  - GAUC     : {metrics.get('gauc', 0.0):.4f}")
                print(f"  - AUC      : {metrics.get('auc', 0.0):.4f}")
                print("  -----------------------------------------------------")
                if total_positive_items > 0:
                    for k in [5, 10, 20, 50]:
                        print(f"  - Recall@{k:<2} : {metrics.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {metrics.get(f'ndcg@{k}', 0.0):.4f}")
                else:
                    print("  - No positive samples for Recall/NDCG in this period.")
                print("  -----------------------------------------------------")
            
        if results_over_periods:
            print(f"\n{'='*20} [ 學習率 {lr} 總結報告 ] {'='*20}")
            avg_metrics = {}
            report_metric_keys = ['gauc', 'auc', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'recall@50', 'ndcg@50']
            for key in report_metric_keys:
                valid_values = [m.get(key) for m in results_over_periods if m.get(key) is not None and np.isfinite(m.get(key))]
                avg_metrics[key] = np.mean(valid_values) if valid_values else 0.0
            print("\n--- Evaluation Finished ---"); print(f"  - GAUC     : {avg_metrics.get('gauc', 0.0):.4f}"); print(f"  - AUC      : {avg_metrics.get('auc', 0.0):.4f}"); print("-------------------------------------------------------")
            for k in [5, 10, 20, 50]:
                print(f"  - Recall@{k:<2} : {avg_metrics.get(f'recall@{k}', 0.0):.4f}   |   NDCG@{k:<2} : {avg_metrics.get(f'ndcg@{k}', 0.0):.4f}")
            print("-------------------------------------------------------")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental Hybrid GNN-MLP Experiment with PyTorch")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get('cuda_visible_DEVICES', "0")
    run_experiment(config)