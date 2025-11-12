import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 匯入你需要的所有工具
from model import LightGCN
from utils import (
    prepare_data_from_dfs,
    BPRDataset,
    build_lightgcn_graph
)

def run_gradient_test(config):
    """
    執行一次性的前向 + 反向傳播，並檢查梯度。
    """
    
    # --- 1. 設置 (和 main.py 相同) ---
    device = torch.device("cpu")
    print(f"--- [DEBUG] Forcing CPU for stability ---")

    print("--- [DEBUG] Loading raw data files ---")
    full_data_df = pd.read_parquet(config['data_path'])
    if config.get('debug_sample', True):
        full_data_df = full_data_df.iloc[::10]
        full_data_df = full_data_df[full_data_df['period'] < 14]
    full_meta_df = pd.read_parquet(config['meta_path'])

    print("--- [DEBUG] Running data preparation ---")
    remapped_full_df, _, _, hyperparams_updates, _, _ = prepare_data_from_dfs(
        full_data_df, full_meta_df, config
    )
    hyperparams = {**config['model'], **hyperparams_updates}
    num_users = hyperparams['num_users']
    num_items = hyperparams['num_items']

    # --- 2. 建立「單一」的圖和模型 (只用 Period 0) ---
    print(f"\n--- [DEBUG] Building graph for Period 0 ---")
    period_id = config['train_start_period']
    current_period_df = remapped_full_df[remapped_full_df['period'] == period_id]
    current_pos_interactions = current_period_df[current_period_df['label'] == 1]
    
    adj_matrix = build_lightgcn_graph(
        current_pos_interactions,
        num_users,
        num_items,
        device
    )

    print(f"--- [DEBUG] Initializing LightGCN model ---")
    model = LightGCN(
        num_users, 
        num_items, 
        hyperparams,
        adj_matrix
    ).to(device)
    
    # (我們不載入任何 checkpoint，使用 Xavier 初始化)
    
    # (我們只拿可訓練的參數)
    optimizer = torch.optim.Adam(model.embeddings.parameters(), lr=0.001)

    # --- 3. 建立 BPR Dataloader ---
    print(f"--- [DEBUG] Building BPRDataset ---")
    train_dataset = BPRDataset(current_pos_interactions, num_items)
    # (我們只跑一個批次)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['model']['batch_size'], 
        shuffle=True
    )

    # --- 4. *** 核心測試：只跑一個批次 *** ---
    print(f"\n{'='*50}")
    print(f"--- [DEBUG] RUNNING SINGLE BATCH TEST ---")
    print(f"{'='*50}")
    
    try:
        # 1. 獲取一個批次
        batch_data = next(iter(train_loader))
        user_id, pos_item_id, neg_item_id = batch_data
        batch_dict = {
            'user_id': user_id.to(device),
            'pos_item_id': pos_item_id.to(device),
            'neg_item_id': neg_item_id.to(device)
        }
        
        print(f"--- [DEBUG] Batch loaded. User shape: {user_id.shape}")

        # 2. 檢查「訓練前」的梯度
        # (這「必須」是 None)
        print(f"\n--- [DEBUG] 1. Gradient BEFORE backward(): ---")
        print(model.embeddings.weight.grad)

        # 3. 執行「完整」的 GNN 訓練步驟
        model.train()
        optimizer.zero_grad()
        
        print(f"\n--- [DEBUG] 2. Running model.forward() (GNN propagation)... ---")
        users_emb, items_emb = model.forward()
        
        print(f"--- [DEBUG] 3. Running model.calculate_loss()... ---")
        loss = model.calculate_loss(users_emb, items_emb, batch_dict)
        
        print(f"--- [DEBUG] 4. Loss reported: {loss.item()} ---")
        # (如果這裡不是 0.6931... 而是 0.0，那可能是 L2 Reg 出了問題)
        if not loss.requires_grad:
            print("\n--- !!! [FATAL] !!! ---")
            print("--- 錯誤：'loss' 張量沒有 'requires_grad=True'。---")
            print("--- 這意味著梯度鏈在某處被 .detach() 了。 ---")
            return

        print(f"--- [DEBUG] 5. Running loss.backward()... ---")
        loss.backward()
        
        print(f"\n--- [DEBUG] 6. Gradient AFTER backward(): ---")
        # 這是我們的「驗收測試」
        grad_tensor = model.embeddings.weight.grad
        
        if grad_tensor is None:
            print("\n--- !!! [TEST FAILED] !!! ---")
            print("--- 錯誤：梯度 (gradient) 是 None。---")
            print("--- 結論：梯度根本沒有回傳到 `model.embeddings`。---")
            print("--- 嫌疑犯：`model.forward()` 或 `model.calculate_loss()` 內部有 .detach() 或 torch.no_grad()。 ---")
        
        elif torch.all(grad_tensor == 0):
            print("\n--- !!! [TEST FAILED] !!! ---")
            print("--- 錯誤：梯度 (gradient) 全是 0.0。---")
            print("--- 結論：梯度流被阻斷了，或者 L2 Reg 是唯一有梯度的部分但太小。---")
        else:
            print("\n--- *** [TEST PASSED] *** ---")
            print(f"--- 梯度成功回傳！ (Sum: {grad_tensor.sum().item()}, Mean: {grad_tensor.mean().item()}) ---")
            print("--- 範例梯度 (前 5 行): ---")
            print(grad_tensor[:5])
            print("\n--- 結論：模型本身 (model.py) 沒問題。問題出在 main.py 的「訓練迴圈」邏輯！ ---")
            
        # 7. (可選) 檢查權重是否更新
        print(f"\n--- [DEBUG] 7. Running optimizer.step()... ---")
        # 儲存舊權重
        old_weights = model.embeddings.weight.data.clone()
        optimizer.step()
        new_weights = model.embeddings.weight.data
        
        if torch.equal(old_weights, new_weights):
            print("--- !!! [OPTIMIZER FAILED] !!! ---")
            print("--- 錯誤：`optimizer.step()` 跑了，但權重沒有任何改變。 ---")
        else:
            print("--- *** [OPTIMIZER PASSED] *** ---")
            print("--- 權重已成功更新！ ---")


    except Exception as e:
        print(f"\n--- [DEBUG] 測試中發生意外錯誤 ---")
        print(e)
        import traceback
        traceback.print_exc()

# --- 和 main.py 一樣的啟動器 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Gradient Debugging Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    
    # 關閉所有 main.py 裡可能會干擾的 torch.no_grad()
    torch.set_grad_enabled(True)
    
    run_gradient_test(config)