import torch
import torch.nn as nn
import optuna
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

try:
    from src.model import DiabetesModel
    from src.preprocess import prepare_data
except ImportError:
    from model import DiabetesModel
    from preprocess import prepare_data

# 目標設定：先確保 Accuracy 至少 0.78，再盡量把 F2 拉高
ACCURACY_FLOOR = 0.78

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
    val_size = int(len(X_train_tensor) * val_ratio)
    train_size = len(X_train_tensor) - val_size
    train_tensor, val_tensor = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    X_train_split, y_train_split = train_tensor.dataset.tensors[0][train_tensor.indices], train_tensor.dataset.tensors[1][train_tensor.indices]
    X_val_split, y_val_split = val_tensor.dataset.tensors[0][val_tensor.indices], val_tensor.dataset.tensors[1][val_tensor.indices]
    return X_train_split, X_val_split, y_train_split, y_val_split


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
    epochs = trial.suggest_int("epochs", 80, 250)
    # 閾值往上移，能減少把太多樣本判成陽性，通常有助於維持 Accuracy
    threshold = trial.suggest_float("threshold", 0.45, 0.75)

    model = DiabetesModel(hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        outputs = model(X_train_split)
        loss = criterion(outputs, y_train_split)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy, precision, recall, f1, f2, _ = evaluate_model(model, X_val_split, y_val_split, threshold=threshold)

    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("threshold", threshold)

    # 如果 Accuracy 沒守住門檻，就直接給很差的分數。
    # 這樣 Optuna 會優先找出「至少夠準」的組合，再去比 F2。
    if accuracy < ACCURACY_FLOOR:
        penalty = (ACCURACY_FLOOR - accuracy) * 10
        return f2 - penalty

    # 只有在 Accuracy 達標時，才把 F2 當主要優化目標。
    return f2

if __name__ == "__main__":
    # n_trials上升，提高Optuna精密程度
    n_trials = 100

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
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])

    for _ in range(best_params["epochs"]):
        best_model.train()
        outputs = best_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    print("\nModel saved successfully!")
