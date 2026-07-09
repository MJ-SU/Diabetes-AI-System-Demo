import json
from pathlib import Path

import optuna
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.model import DiabetesModel
    from src.preprocess import prepare_data
except ImportError:
    from model import DiabetesModel
    from preprocess import prepare_data

# 目標設定：先確保 Accuracy 至少 0.78，再盡量把 F2 拉高
ACCURACY_FLOOR = 0.75

#定義評估模型函數(計算準確率、精確率、召回率、F1分數、F2分數和混淆矩陣)
def evaluate_model(model, X_tensor, y_tensor, threshold=0.5):
    model.eval()  #設置模型為評估模式
    with torch.no_grad():
        outputs = model(X_tensor).squeeze(1)
        # threshold 越低，模型越容易判成陽性，Recall 通常會上升，但 Precision 可能下降
        predicted = (outputs > threshold).to(torch.int64)  #將輸出轉換為0或1
        y_true = y_tensor.squeeze(1).cpu().numpy().astype(int)  #將y_tensor轉換為numpy陣列
        y_pred = predicted.cpu().numpy().astype(int)  #將predicted轉換為numpy陣列
        #計算各種評估指標
        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred, zero_division=0)

        recall = recall_score(y_true, y_pred, zero_division=0)

        f1 = f1_score(y_true, y_pred, zero_division=0)

        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, f2, cm 


def split_train_val(X_train_tensor, y_train_tensor, val_ratio=0.2):
    # 這裡切出驗證集給 Optuna 用；訓練完成後，測試集只拿來做最後一次正式評估
    indices = np.arange(len(X_train_tensor))
    stratify_labels = y_train_tensor.squeeze(1).cpu().numpy().astype(int)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=42,
        stratify=stratify_labels,
    )
    X_train_split = X_train_tensor[train_indices]
    y_train_split = y_train_tensor[train_indices]
    X_val_split = X_train_tensor[val_indices]
    y_val_split = y_train_tensor[val_indices]
    return X_train_split, X_val_split, y_train_split, y_val_split


def compute_positive_weight(y_tensor):
    y_np = y_tensor.squeeze(1).cpu().numpy().astype(int)
    positive_count = max(int((y_np == 1).sum()), 1)
    negative_count = max(int((y_np == 0).sum()), 1)
    raw_weight = np.sqrt(negative_count / positive_count)
    return torch.tensor(min(raw_weight, 1.5), dtype=torch.float32)


def train_model(model, X_train_split, y_train_split, learning_rate, epochs):
    dataset = TensorDataset(X_train_split, y_train_split)
    batch_size = min(32, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss(reduction="none")
    pos_weight = compute_positive_weight(y_train_split)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            raw_loss = criterion(outputs, batch_y)
            sample_weight = torch.ones_like(batch_y) + (pos_weight.to(batch_y.device) - 1.0) * batch_y
            loss = (raw_loss * sample_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_one_trial(trial):
    # 每個 trial 都重新讀資料，確保搜尋參數時的比較基準一致
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = prepare_data()

    X_train_split, X_val_split, y_train_split, y_val_split = split_train_val(X_train_tensor, y_train_tensor)

    # - hidden1 / hidden2：模型容量
    # - dropout：防止過擬合
    # - learning_rate：學習速度
    # - epochs：訓練輪數
    # - threshold：最後怎麼把機率切成 0 / 1
    hidden1 = trial.suggest_categorical("hidden1", [8, 16, 32, 64])
    hidden2 = trial.suggest_categorical("hidden2", [4, 8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    epochs = trial.suggest_int("epochs", 80, 150)
    # F2 比 F1 更重視 recall，所以閾值不應只往高區間找；
    # 這裡把候選範圍往低閾值移，避免模型過度保守。
    threshold = trial.suggest_float("threshold", 0.25, 0.55)

    model = DiabetesModel(hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    model = train_model(model, X_train_split, y_train_split, learning_rate, epochs)

    accuracy, precision, recall, f1, f2, _ = evaluate_model(model, X_val_split, y_val_split, threshold=threshold)
    balanced_score = (0.6 * accuracy) + (0.4 * f2)

    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("threshold", threshold)
    trial.set_user_attr("balanced_score", balanced_score)

    # 如果 Accuracy 沒守住門檻，就直接給很差的分數。
    # 這樣 Optuna 會優先找出「至少夠準」的組合，再去比 F2。
    if accuracy < ACCURACY_FLOOR:
        penalty = (ACCURACY_FLOOR - accuracy) * 10
        return balanced_score - penalty

    # Accuracy 與 F2 都有表現時，優先選擇更均衡、但不犧牲太多準確率的組合。
    return balanced_score

if __name__ == "__main__":
<<<<<<< HEAD
    # n_trials上升，提高Optuna精密程度
    n_trials = 100
=======
    # 讓 Optuna 更仔細找，可以把 n_trials 從 20 提高到 50、100
    # 這會更慢，但通常更有機會找到更好的組合
    n_trials = 80
>>>>>>> 94e0fec (refine training balance and prediction defaults)

    study = optuna.create_study(direction="maximize")
    study.optimize(train_one_trial, n_trials=n_trials)

    print("Best trial:")
    print(f"  F2 Score: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    best_params = study.best_params
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = prepare_data(save_scaler_to_disk=True)

    best_model = DiabetesModel(
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout=best_params["dropout"],
    )
    best_model = train_model(
        best_model,
        X_train_tensor,
        y_train_tensor,
        best_params["learning_rate"],
        best_params["epochs"],
    )

    accuracy, precision, recall, f1, f2, cm = evaluate_model(
        best_model,
        X_test_tensor,
        y_test_tensor,
        threshold=best_params["threshold"],
    )

    print(f"Best Threshold: {best_params['threshold']:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print("Confusion Matrix:")
    print(cm)
    #儲存訓練好的模型
    model_path = Path(__file__).resolve().parent.parent / "models"
    model_path.mkdir(exist_ok=True)

    torch.save(
        best_model.state_dict(),
        model_path / "diabetes_model.pth"
    )

<<<<<<< HEAD
    print("\nModel saved successfully!")
=======
    training_meta = {
        "best_threshold": best_params["threshold"],
        "best_params": best_params,
        "test_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
        },
    }

    (model_path / "training_meta.json").write_text(
        json.dumps(training_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nModel saved successfully!")
    print(f"Training metadata saved to: {model_path / 'training_meta.json'}")
>>>>>>> 94e0fec (refine training balance and prediction defaults)
