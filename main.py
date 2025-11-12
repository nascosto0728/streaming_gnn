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
import torch.nn.functional as F # (NEW) 為了 Val Loss
from torch.utils.data import DataLoader
# (log_loss 暫時用不到，BPR loss 在模型內)

# --- (*** 核心修改 1: 匯入新工具 ***) ---
from model import LightGCN # (NEW) 匯入 GNN
from utils import (
    prepare_data_from_dfs,
    RecommendationDataset, # (保留) 為了 Val/Test
    sample_negative_users, # (保留) 為了 Test
    BPRDataset,            # (NEW) 為了 Train
    build_lightgcn_graph   # (NEW) 為了建圖
)
import random


def run_experiment(config: Dict[str, Any]):
    """
    主實驗執行函式 (*** GNN 重構版 ***)。
    """
    # --- 0. 設備設定 ---
    device = torch.device("cpu") # GNN 在 CPU 上通常更穩定
    if torch.backends.mps.is_available():
        device = torch.device("cpu")
        print("--- MPS is available! Using MPS. ---")
    else:
        print("--- MPS not found. Using CPU. ---")

    random.seed(42)
    # --- 1. 數據 I/O ---
    print("--- Loading raw data files ---")
    start_time = time.time()
    
    full_data_df = pd.read_parquet(config['data_path'])
    if config.get('debug_sample', True):
        # full_data_df = full_data_df.iloc[::10]
        full_data_df = full_data_df[full_data_df['period'] < 14]
    full_meta_df = pd.read_parquet(config['meta_path'])

    # --- 2. 數據轉換 ---
    # (MODIFIED) GNN 不需要 cates, cate_lens, item_map, cate_map
    #            但我們「保留」它們是為了舊的評估邏輯
    remapped_full_df, cates, cate_lens, hyperparams_updates, item_map, full_cate_map = prepare_data_from_dfs(
        full_data_df, full_meta_df, config
    )
    hyperparams = {**config['model'], **hyperparams_updates}
    config['model'] = hyperparams
    num_users = hyperparams['num_users']
    num_items = hyperparams['num_items']
    print(f"Global data processing finished. (Users: {num_users}, Items: {num_items})")
    
    # --- (*** 核心修改 2: 刪除向量載入 ***) ---
    # (GNN 預設使用 Xavier 初始化，我們刪除所有 SBERT/KGE 的 .npy 載入邏輯)
    print("--- GNN mode: Skipping pre-loading of text vectors. ---")

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
        seen_users_pool = set()
        
        # (NEW) GNN 需要一個「不斷增長」的互動 DataFrame
        incremental_interaction_df = pd.DataFrame()

        # --- 4. 進入增量訓練主迴圈 ---
        for period_id in range(config['train_start_period'], config['num_periods']):
            print(f"\n{'='*25} Period {period_id} {'='*25}")
            
            # --- (*** 核心修改 3: GNN 資料準備 ***) ---
            
            # 1. 獲取「當前時期」的資料
            current_period_df = remapped_full_df[remapped_full_df['period'] == period_id]
            
            # 2. 獲取測試集 (邏輯不變)
            test_set = None
            if period_id >= config['test_start_period'] and (period_id + 1) < config['num_periods']:
                test_set = remapped_full_df[remapped_full_df['period'] == period_id + 1].iloc[::10]

            if current_period_df.empty:
                print(f"No training data for period {period_id}. Skipping.")
                continue
                
            # 3. 更新「全歷史」互動 (GNN 建圖用)
            # 我們只關心正互動 (label=1)
            current_pos_interactions = current_period_df[current_period_df['label'] == 1]
            if incremental_interaction_df.empty:
                incremental_interaction_df = current_pos_interactions
            else:
                incremental_interaction_df = pd.concat(
                    [incremental_interaction_df, current_pos_interactions],
                    ignore_index=True
                )
            # (去重，以防萬一)
            incremental_interaction_df = incremental_interaction_df.drop_duplicates(subset=['userId', 'itemId'])
            
            # 4. 建立「增量圖」
            # GNN 在每個 period 都會在「至今為止」的「全圖」上訓練
            adj_matrix = build_lightgcn_graph(
                incremental_interaction_df,
                num_users,
                num_items,
                device
            )

            # 5. 更新評估用的 history (邏輯不變)
            print(f"Incrementally updating item history and seen users pool...")
            current_period_interactions_dict = current_period_df.groupby('itemId')['userId'].apply(set).to_dict()
            for item_id, user_set in current_period_interactions_dict.items():
                item_history_dict.setdefault(item_id, set()).update(user_set)
            seen_users_pool.update(current_period_df['userId'].unique())
                
            # --- (*** 核心修改 4: GNN DataLoaders ***) ---
            
            # 1. 獲取早停參數 (邏輯不變)
            val_split_ratio = config.get('validation_split', 0.2)
            
            # 2. 切分「當前時期」的「訓練集」和「驗證集」(邏輯不變)
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
            # (NEW) 訓練集使用 BPRDataset
            #   (注意: BPRDataset 只關心正樣本，所以我們傳入 train_set)
            train_pos_set = train_set[train_set['label'] == 1]
            train_dataset = BPRDataset(train_pos_set, num_items)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['model']['batch_size'], 
                shuffle=True, # BPR 必須 shuffle
                num_workers=config.get('num_workers', 0)
            )
            
            # (MODIFIED) 驗證集使用 RecommendationDataset
            #   (我們需要 (u, i, label) 來計算 BCE Loss)
            val_loader = None
            if val_set is not None and not val_set.empty:
                val_dataset = RecommendationDataset(val_set, {})
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config['model']['batch_size'],
                    shuffle=False, 
                    num_workers=config.get('num_workers', 0)
                )


            # --- (*** 核心修改 5: GNN 模型初始化 ***) ---
            model = LightGCN(
                num_users, 
                num_items, 
                hyperparams,
                adj_matrix # 傳入「當前」的圖
            ).to(device)
            
            # (MODIFIED) GNN 只訓練 Embeddings
            optimizer = torch.optim.Adam(model.embeddings.parameters(), lr=lr)
            
            # --- (*** 核心修改 6: GNN 權重恢復 ***) ---
            # 我們只恢復「Embeddings」，不恢復「圖」
            best_model_path = '' # (重置)
            if period_id > config['train_start_period']:
                prev_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id-1}')
                ckpt_path = os.path.join(prev_ckpt_dir, 'best_embeddings.pth') # <--- 只載入 .pth
                if os.path.exists(ckpt_path):
                    print(f"Attempting to restore model weights from: {ckpt_path}")
                    try:
                        # (MODIFIED) 只載入 embedding weight
                        model.embeddings.weight.data = torch.load(ckpt_path, map_location=device)
                        print("--- Successfully restored model EMBEDDINGS. ---")
                    except Exception as e:
                        print(f"Could not load embedding weights: {e}")

            # --- (*** 核心修改 7: GNN 訓練迴圈 ***) ---
            
            # 1. 初始化早停參數
            max_epochs = config.get('max_epochs', 100)
            patience = config.get('patience', 10)
            patience_counter = 0
            best_val_loss = float('inf')
            
            period_ckpt_dir = os.path.join('./checkpoints', dir_name_with_lr, f'period_{period_id}')
            os.makedirs(period_ckpt_dir, exist_ok=True)
            # (MODIFIED) 我們只存 Embeddings
            best_model_path = os.path.join(period_ckpt_dir, 'best_embeddings.pth')

            print(f"--- Starting GNN training loop (Max Epochs: {max_epochs}, Patience: {patience}) ---")
            
            pbar_epochs = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=True)
            for epoch_id in pbar_epochs:
                start_time_epoch = time.time()
                
                # --- (a) 訓練階段 ---
                model.train()
                
                losses = []
                for batch_data in train_loader:
                    # (NEW) BPRDataset 回傳的是 (u, i_pos, i_neg)
                    user_id, pos_item_id, neg_item_id = batch_data
                    batch_dict = {
                        'user_id': user_id.to(device),
                        'pos_item_id': pos_item_id.to(device),
                        'neg_item_id': neg_item_id.to(device)
                    }
                    
                    optimizer.zero_grad()
                    # (NEW) GNN 的 loss 是在「查找」之後算的
                    # GNN "前向傳播" (K層)
                    # 這必須在迴圈內，以建立一個「乾淨」的計算圖
                    users_emb, items_emb = model.forward()
                    
                    # BPR Loss
                    loss = model.calculate_loss(users_emb, items_emb, batch_dict)
                    
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()

                    losses.append(loss.item())
                avg_train_loss = np.mean(losses)
                
                # --- (b) 驗證階段 ---
                if val_loader is None:
                    print(f"  Epoch {epoch_id} Avg Train BPR Loss: {avg_train_loss:.4f} (No validation set)")
                    torch.save(model.embeddings.weight.data, best_model_path)
                    continue

                model.eval()
                val_losses = []
                
                # GNN: 驗證時，也先執行一次「全圖」前向傳播
                # (這在 eval 模式下是 OK 的，因為 no_grad() 會關閉圖的建立)
                with torch.no_grad():
                    val_users_emb, val_items_emb = model.forward()
                
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        u_emb = val_users_emb[batch['users']]
                        i_emb = val_items_emb[batch['items']]
                        scores = torch.sum(u_emb * i_emb, dim=1)
                        labels = batch['labels'].float()
                        val_loss = F.binary_cross_entropy_with_logits(scores, labels).item()
                        val_losses.append(val_loss)
                        
                        
                avg_val_loss = np.mean(val_losses)

                # --- (c) 決策階段 (Early Stopping) ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # (MODIFIED) *** 只儲存 Embeddings ***
                    torch.save(model.embeddings.weight.data, best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"--- Early stopping triggered after {epoch_id} epochs. ---")
                    break 

            
                pbar_epochs.set_postfix(train_bpr=f"{avg_train_loss:.4f}", val_bce=f"{avg_val_loss:.4f}", patience=f" {patience_counter}/{patience}")
            
            print(f"--- Training finished for period {period_id}. Best embeddings saved to {best_model_path} ---")

            # --- (*** 核心修改 8: GNN 評估迴圈 ***) ---
            if test_set is not None and not test_set.empty:
                seen_users_pool.update(test_set['userId'].unique())
                
                if os.path.exists(best_model_path):
                    print(f"--- Loading BEST embeddings from {best_model_path} for evaluation ---")
                    model.embeddings.weight.data = torch.load(best_model_path, map_location=device)
                else:
                    print(f"--- WARNING: No best embeddings found. Evaluating with last-epoch model. ---")
                
                model.eval() 
                
                # (NEW) GNN: 評估前，執行最終的「全圖」前向傳播
                print("--- GNN: Running full-graph forward pass for evaluation ---")
                with torch.no_grad():
                    eval_users_emb, eval_items_emb = model.forward()
                print("--- GNN: Embeddings are ready ---")

                print(f"\n--- Running Unified Sampled Evaluation ---")
                
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
                        for batch in tqdm(test_loader_pos, desc="Evaluating (GNN)"):
                            batch_cpu = {k: v for k, v in batch.items()}
                            batch = {k: v.to(device) for k, v in batch.items()}
                            batch_size = len(batch['users'])

                            # 1. 負採樣 
                            neg_user_ids_list = []
                            for i in range(batch_size):
                                item_i_raw = batch_cpu['items_raw'][i].item()
                                user_j_id = batch['users'][i].item()
                                seen_users = item_history_dict.get(item_i_raw, set())
                                neg_users = sample_negative_users(
                                    user_pool=seen_users_pool,
                                    seen_users_set=seen_users,
                                    positive_user_id=user_j_id,
                                    num_samples=sampling_size,
                                    device=device
                                )
                                neg_user_ids_list.append(neg_users)
                            neg_user_ids_batch = torch.stack(neg_user_ids_list) 

                            # 2. (MODIFIED) 調用 GNN 的 inference
                            #    我們不再需要 model.inference，因為 model.py 裡的新函式
                            #    inference_for_evaluation 已經準備好了
                            pos_logits, neg_logits = model.inference_for_evaluation(
                                eval_users_emb, 
                                eval_items_emb, 
                                batch, 
                                neg_user_ids_batch
                            )
                            
                            # 3. 組合 logits (邏輯 100% 不變)
                            pos_logits = pos_logits.unsqueeze(1) 
                            all_logits = torch.cat([pos_logits, neg_logits], dim=1) 
                            
                            # 4. 排名 (邏輯 100% 不變)
                            ranks = (all_logits > pos_logits).sum(dim=1).cpu().numpy()

                            # 5. AUC (邏輯 100% 不變)
                            auc_per_sample = (pos_logits > neg_logits).float().mean(dim=1)
                            all_per_sample_aucs.extend(auc_per_sample.cpu().numpy())
                            all_user_ids_for_gauc.extend(batch_cpu['users_raw'].numpy())

                            # 6. 累加 (邏輯 100% 不變)
                            for rank in ranks:
                                for j, k in enumerate(Ks):
                                    if rank < k:
                                        total_recalls[j] += 1
                                        total_ndcgs[j] += 1 / np.log2(rank + 2)
                
                # --- (後面所有指標計算、報告的邏輯 100% 不變) ---
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
            
        # (最終總結報告邏輯 100% 不變)
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
    parser = argparse.ArgumentParser(description="Incremental GNN Learning Experiment with PyTorch")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get('cuda_visible_devices', "0")
    run_experiment(config)