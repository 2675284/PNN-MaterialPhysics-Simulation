# evaluate.py - 测试集评价文件
import torch
from model import PINN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

def evaluate_model():
    # 硬件配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = PINN(input_dim=3, hidden_dim=128).to(device)

    # 加载最好的模型
    checkpoint = torch.load("model/best_pinn_model_epoch_2000.pth")
    model.load_state_dict(checkpoint['model'])

    # 加载测试集数据
    test_data_path = 'data/test_data.pth'
    if os.path.exists(test_data_path):
        test_data = torch.load(test_data_path)
        X_test = test_data['x_test']
        y_test = test_data['y_test']
    else:
        raise FileNotFoundError(f"测试集数据文件 {test_data_path} 未找到。")

    # 将数据转换为张量并移到设备上
    if isinstance(X_test, torch.Tensor):
        X_test_tensor = X_test.clone().detach().to(device)
    else:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    if isinstance(y_test, torch.Tensor):
        y_test_tensor = y_test.clone().detach().to(device)
    else:
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 在测试集上进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)

    # 将预测结果和目标值移到CPU并转换为numpy数组
    predictions = outputs.cpu().numpy()
    targets = y_test_tensor.cpu().numpy()

    # 计算评价指标
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    print(f"Mean Squared Error on Test Set: {mse}")
    print(f"Mean Absolute Error on Test Set: {mae}")
    print(f"Root Mean Squared Error on Test Set: {rmse}")
    print(f"R-squared Score on Test Set: {r2}")

if __name__ == "__main__":
    evaluate_model()