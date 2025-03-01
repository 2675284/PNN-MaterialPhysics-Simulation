# data_loader.py
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class PNDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_df = self._load_raw_data()
        self._parse_columns()
        self._init_normalization()
        self._save_processed_data()  # 调用保存方法
        self._split_train_test()  # 调用划分数据集方法

    def _init_normalization(self):
        """使用PyTorch计算均值和标准差"""
        data = torch.tensor(self.data, dtype=torch.float32)

        # 输入归一化 (X, Y, V_bias)
        self.x_mean = data[:, :3].mean(dim=0)
        self.x_std = data[:, :3].std(dim=0)

        # 输出归一化 (V, N)
        self.y_mean = data[:, 3:5].mean(dim=0)
        self.y_std = data[:, 3:5].std(dim=0)

        # 归一化数据
        self.x_data = (data[:, :3] - self.x_mean) / self.x_std
        self.y_data = (data[:, 3:5] - self.y_mean) / self.y_std

    def inverse_normalize_x(self, x_norm):
        """反归一化输入"""
        return x_norm * self.x_std.to(x_norm.device) + self.x_mean.to(x_norm.device)

    def inverse_normalize_y(self, y_norm):
        """反归一化输出"""
        return y_norm * self.y_std.to(y_norm.device) + self.y_mean.to(y_norm.device)

    def _load_raw_data(self):
        """加载CSV文件并清理列名"""
        try:
            df = pd.read_csv(self.file_path, skiprows=8, engine='python')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_path, skiprows=8, encoding='gbk', engine='python')

        # 列名清洗逻辑（保留关键标识符）
        df.columns = (
            df.columns.str.strip()
            .str.replace(r'\s+', '', regex=True)  # 移除所有空格
            .str.replace(r'\(', '', regex=True)  # 移除左括号
            .str.replace(r'\)', '', regex=True)  # 移除右括号
            .str.replace(r'@\s*V_bias=', '@V_bias=', regex=True)  # 统一格式
        )
        print("处理后的列名:", df.columns.tolist())
        return df

    def _parse_columns(self):
        """动态解析电压和浓度列"""
        self.samples = []

        # 提取所有电压相关列
        voltage_cols = [col for col in self.raw_df.columns if '@V_bias=' in col]

        # 按电压值分组
        voltage_groups = {}
        for col in voltage_cols:
            # 正则匹配列名格式（示例：VV@V_bias=0 或 semi.N1/m^3@V_bias=0）
            match = re.match(r'([A-Za-z0-9.]+/m\^3)@V_bias=([0-9.]+)|([A-Za-z0-9.]+)@V_bias=([0-9.]+)', col)
            if match:
                if match.group(1):  # 匹配浓度列：semi.N1/m^3@V_bias=0
                    col_type = 'semi.N'
                    v_bias = float(match.group(2))
                else:  # 匹配电压列：VV@V_bias=0
                    col_type = 'V'
                    v_bias = float(match.group(4))

                if v_bias not in voltage_groups:
                    voltage_groups[v_bias] = {}
                voltage_groups[v_bias][col_type] = col

        # 检查是否解析到有效列
        if not voltage_groups:
            raise ValueError("未找到有效的电压列，请检查列名格式（示例：VV@V_bias=0）")

        # 构建样本数据
        for idx, row in self.raw_df.iterrows():
            x = row['X']
            y = row['Y']
            for v_bias, cols in voltage_groups.items():
                try:
                    v_value = row[cols['V']]  # 电压值列
                    n_value = row[cols['semi.N']]  # 浓度列
                except KeyError as e:
                    raise KeyError(f"列名 {e} 不存在，实际列名: {cols}") from None
                self.samples.append([x, y, v_bias, v_value, n_value])

        self.data = np.array(self.samples, dtype=np.float32)

    def _normalize(self):
        """数据标准化"""
        self.x_scaler = StandardScaler()
        self.x_data = self.x_scaler.fit_transform(self.data[:, :3])

        self.y_scaler = StandardScaler()
        self.y_data = self.y_scaler.fit_transform(self.data[:, 3:5])

    def get_dataloader(self, batch_size=1024):
        """生成DataLoader"""
        dataset = PNDataset(self.x_data, self.y_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_train_dataloader(self, batch_size=1024):
        """生成训练集DataLoader"""
        train_data = torch.load('data/train_data.pth')
        X_train = train_data['x_train']
        y_train = train_data['y_train']
        dataset = PNDataset(X_train, y_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_test_dataloader(self, batch_size=1024):
        """生成测试集DataLoader"""
        test_data = torch.load('data/test_data.pth')
        X_test = test_data['x_test']
        y_test = test_data['y_test']
        dataset = PNDataset(X_test, y_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _save_processed_data(self):
        """保存处理后的数据到data文件夹"""
        # 检查data文件夹是否存在，不存在则创建
        if not os.path.exists('data'):
            os.makedirs('data')
        processed_data = {
            'x_data': self.x_data,
            'y_data': self.y_data,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std
        }
        torch.save(processed_data, 'data/processed_data.pth')

    def load_processed_data(self):
        """加载处理后的数据"""
        if os.path.exists('data/processed_data.pth'):
            processed_data = torch.load('data/processed_data.pth')
            self.x_data = processed_data['x_data']
            self.y_data = processed_data['y_data']
            self.x_mean = processed_data['x_mean']
            self.x_std = processed_data['x_std']
            self.y_mean = processed_data['y_mean']
            self.y_std = processed_data['y_std']

    def _split_train_test(self):
        """划分训练集和测试集，并保存到data文件夹"""
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size=0.2, random_state=42)

        # 检查data文件夹是否存在，不存在则创建
        if not os.path.exists('data'):
            os.makedirs('data')

        # 保存训练集
        train_data = {
            'x_train': X_train,
            'y_train': y_train
        }
        torch.save(train_data, 'data/train_data.pth')

        # 保存测试集
        test_data = {
            'x_test': X_test,
            'y_test': y_test
        }
        torch.save(test_data, 'data/test_data.pth')


class PNDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data.clone().detach() if isinstance(x_data, torch.Tensor) else torch.tensor(x_data,
                                                                                                    dtype=torch.float32)
        self.y_data = y_data.clone().detach() if isinstance(y_data, torch.Tensor) else torch.tensor(y_data,
                                                                                                    dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]