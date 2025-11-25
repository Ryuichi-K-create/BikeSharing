import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.features import BikeSharingPreprocessor, get_target
from src.models import BikeSharingMLP
import itertools

# 設定
RANDOM_STATE = 42
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_mlp(X_train, y_train, X_val, y_val, params):
    # パラメータ展開
    hidden_layer_sizes = params['hidden_layer_sizes']
    activation = params['activation']
    alpha = params['alpha']
    lr = params['learning_rate_init']
    max_iter = params.get('max_iter', 500)
    batch_size = 64

    # データセット作成
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    # val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1)) # Not used in loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # モデル初期化
    input_dim = X_train.shape[1]
    model = BikeSharingMLP(input_dim, hidden_layer_sizes, activation)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    no_improve = 0

    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

    for epoch in range(max_iter):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss, train_losses, val_losses

def main():
    set_seed(RANDOM_STATE)
    
    # 1. データ読み込み
    print("Loading data...")
    # ファイル名が train.csv であることに注意
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    # 2. 前処理
    print("Preprocessing data...")
    preprocessor = BikeSharingPreprocessor()
    X = preprocessor.fit_transform(train_df)
    y = get_target(train_df).values # numpy array

    X_test_submit = preprocessor.transform(test_df)

    # 3. データ分割 (時系列を考慮してシャッフルなしで分割)
    # Train: 60%, Val: 20%, Test: 20%
    # 時系列データのリークを防ぐため、shuffle=False とする
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=False)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    if preprocessor.feature_names is not None:
        print(f"Feature names example: {preprocessor.feature_names[:10]} ...")

    # ログ用リスト
    experiment_logs = []

    # 4. モデル学習 & 評価

    # --- Linear Regression (Ridge with Tuning) ---
    print("Training Linear Regression (Ridge)...")
    
    ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, ridge_params, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_train, y_train)
    
    best_ridge = ridge_cv.best_estimator_
    print(f"Best Ridge Params: {ridge_cv.best_params_}")
    
    # Valでの評価
    y_pred_val_ridge = best_ridge.predict(X_val)
    rmsle_val_ridge = np.sqrt(mean_squared_error(y_val, y_pred_val_ridge))
    print(f"Ridge Val RMSLE: {rmsle_val_ridge:.4f}")
    
    # Testでの評価
    y_pred_test_ridge = best_ridge.predict(X_test)
    rmsle_test_ridge = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge))
    print(f"Ridge Test RMSLE: {rmsle_test_ridge:.4f}")

    experiment_logs.append({
        'Model': 'Ridge Regression',
        'Parameters': str(ridge_cv.best_params_),
        'Val RMSLE': rmsle_val_ridge,
        'Test RMSLE': rmsle_test_ridge
    })

    # --- MLP Regressor (PyTorch) ---
    print("Training MLP Regressor (PyTorch)...")
    
    # パラメータ探索空間
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_mlp_model = None
    best_mlp_rmsle = float('inf')
    best_mlp_params = None
    best_train_losses = None
    best_val_losses = None
    
    print(f"Starting Grid Search with {len(param_combinations)} combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"[{i+1}/{len(param_combinations)}] Testing params: {params}")
        model, val_loss, train_losses, val_losses = train_mlp(X_train, y_train, X_val, y_val, params)
        
        val_rmsle = np.sqrt(val_loss)
        print(f"  -> Val RMSLE: {val_rmsle:.4f}")
        
        if val_rmsle < best_mlp_rmsle:
            best_mlp_rmsle = val_rmsle
            best_mlp_model = model
            best_mlp_params = params
            best_train_losses = train_losses
            best_val_losses = val_losses

    print(f"Best MLP Val RMSLE: {best_mlp_rmsle:.4f}")
    print(f"Best Params: {best_mlp_params}")

    # Testでの評価 (Best MLP)
    best_mlp_model.eval()
    with torch.no_grad():
        y_pred_test_mlp_tensor = best_mlp_model(torch.FloatTensor(X_test))
        y_pred_test_mlp = y_pred_test_mlp_tensor.numpy().flatten()
    
    rmsle_test_mlp = np.sqrt(mean_squared_error(y_test, y_pred_test_mlp))
    print(f"MLP Test RMSLE: {rmsle_test_mlp:.4f}")

    experiment_logs.append({
        'Model': 'MLP Regressor (PyTorch)',
        'Parameters': str(best_mlp_params),
        'Val RMSLE': best_mlp_rmsle,
        'Test RMSLE': rmsle_test_mlp
    })

    # 5. 結果保存

    # ログ保存
    log_df = pd.DataFrame(experiment_logs)
    log_df.to_csv(os.path.join(OUTPUT_DIR, 'experiment_log.csv'), index=False)
    print(f"Experiment log saved to {os.path.join(OUTPUT_DIR, 'experiment_log.csv')}")

    # 可視化: 実測値 vs 予測値 (Test Set Comparison)
    plt.figure(figsize=(10, 6))
    
    # Ridge
    plt.scatter(y_test, y_pred_test_ridge, alpha=0.3, label=f'Ridge (RMSLE={rmsle_test_ridge:.4f})', color='blue')
    # MLP
    plt.scatter(y_test, y_pred_test_mlp, alpha=0.3, label=f'MLP (RMSLE={rmsle_test_mlp:.4f})', color='orange')
    
    # 理想線
    min_val = min(y_test.min(), y_pred_test_ridge.min(), y_pred_test_mlp.min())
    max_val = max(y_test.max(), y_pred_test_ridge.max(), y_pred_test_mlp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    plt.xlabel('Actual (log1p scale)')
    plt.ylabel('Predicted (log1p scale)')
    plt.title('Actual vs Predicted on Test Set (Ridge vs MLP)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_comparison.png'))
    plt.close()
    print(f"Prediction comparison plot saved to {os.path.join(OUTPUT_DIR, 'prediction_comparison.png')}")

    # --- 追加の可視化: 時系列での予測比較 (最初の1週間 = 168時間) ---
    # Testデータのインデックスをリセットしていないため、プロット用に抽出
    # y_test は numpy array ではない可能性があるため確認
    if not isinstance(y_test, np.ndarray):
        y_test_np = y_test
    else:
        y_test_np = y_test

    subset_n = 168 # 1週間分
    
    # Ridge Time Series
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_np[:subset_n], label='Actual', color='black', alpha=0.7)
    plt.plot(y_pred_test_ridge[:subset_n], label='Ridge Prediction', color='blue', alpha=0.7)
    plt.title('Time Series Comparison: Actual vs Ridge (First 1 Week of Test Set)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Count (log1p scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'timeseries_ridge.png'))
    plt.close()
    
    # MLP Time Series
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_np[:subset_n], label='Actual', color='black', alpha=0.7)
    plt.plot(y_pred_test_mlp[:subset_n], label='MLP Prediction', color='orange', alpha=0.7)
    plt.title('Time Series Comparison: Actual vs MLP (First 1 Week of Test Set)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Count (log1p scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'timeseries_mlp.png'))
    plt.close()
    print("Time series plots saved.")

    # 可視化: Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(best_train_losses, label='Training Loss')
    plt.plot(best_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('MLP Training & Validation Loss Curve (Best Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()
    print(f"Loss curve saved to {os.path.join(OUTPUT_DIR, 'loss_curve.png')}")

    # Submission作成
    # テストデータに対する予測
    best_mlp_model.eval()
    with torch.no_grad():
        y_pred_submit_log_tensor = best_mlp_model(torch.FloatTensor(X_test_submit))
        y_pred_submit_log = y_pred_submit_log_tensor.numpy().flatten()
        
    # 対数スケールから戻す
    y_pred_submit = np.expm1(y_pred_submit_log)
    # 負の値は0にする
    y_pred_submit = np.maximum(y_pred_submit, 0)

    submission = pd.DataFrame({
        'datetime': test_df['datetime'],
        'count': y_pred_submit
    })
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"Submission saved to {os.path.join(OUTPUT_DIR, 'submission.csv')}")

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
