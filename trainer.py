import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer:
    def __init__(self, data_loader, model, device, params):
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.params = params
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, start_epoch, num_epochs, total_epochs):
        if num_epochs <= 0:
            return None

        total_loss = 0
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            train_dataloader = self.data_loader.get_train_dataloader()
            epoch_loss = 0

            # 使用 tqdm 包装数据加载器，显示进度条
            with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{total_epochs}', unit='batch') as pbar:
                for batch_x, batch_y in pbar:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # 前向传播
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    # 更新进度条的显示信息
                    pbar.set_postfix({'Loss': loss.item()})

            total_loss += epoch_loss
            print(f'Epoch {epoch + 1}/{total_epochs}: Loss = {epoch_loss}')

        return total_loss if total_loss > 0 else None