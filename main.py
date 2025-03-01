# main.py - 主程序
from config import MaterialParameters
from data_loader import PNDataLoader
from model import PINN
from trainer import Trainer
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

if __name__ == "__main__":
    # 硬件配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化材料参数
    params = MaterialParameters()

    # 加载数据
    data_loader = PNDataLoader("pn_junction.csv")

    # 初始化模型
    model = PINN(input_dim=3, hidden_dim=128).to(device)

    # 训练配置
    trainer = Trainer(data_loader, model, device, params)  # 调整参数顺序

    # 执行训练
    total_epochs = 2000
    best_loss = float('inf')
    best_model = None

    start_epoch = 0
    while start_epoch < total_epochs:
        num_epochs_to_train = min(100, total_epochs - start_epoch)
        total_loss = trainer.train(start_epoch, num_epochs_to_train, total_epochs)
        if total_loss is None:
            print(f"Epoch {start_epoch + num_epochs_to_train}: Loss is None, skipping model saving.")
            start_epoch += num_epochs_to_train
            continue

        # 保存当前最好的模型
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = model.state_dict()
            if not os.path.exists('model'):
                os.makedirs('model')
            torch.save({
                'model': best_model
            }, f"model/best_pinn_model_epoch_{start_epoch + num_epochs_to_train}.pth")

        # 保存当前的最后模型
        torch.save({
            'model': model.state_dict()
        }, f"model/trained_pinn_model_epoch_{start_epoch + num_epochs_to_train}.pth")

        start_epoch += num_epochs_to_train

    # 训练完成后，使用测试集进行评价
    print("Training completed. Evaluating on the test set...")
    test_dataloader = data_loader.get_test_dataloader()
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # 计算评价指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Mean Squared Error on Test Set: {mse}")
    print(f"Mean Absolute Error on Test Set: {mae}")