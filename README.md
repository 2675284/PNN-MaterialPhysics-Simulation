# PNN-MaterialPhysics-Simulation

这是一个基于物理信息神经网络（PNN）的材料物理模拟项目，主要用于训练模型并对测试集进行评估。项目包含了数据加载、模型定义、训练和评估等功能模块,训练数据使用的COMSOL建模简易二维pn结的电势和电子浓度数据。

## 项目结构
```
/
├── config.py          # 物理常数与材料参数配置文件
├── data_loader.py     # 数据加载模块
├── evaluate.py        # 测试集评价文件
├── model.py           # 模型定义文件
├── physics.py         # 物理相关功能文件
├── trainer.py         # 模型训练模块
├── data/              # 数据文件夹，用于存放训练、测试数据及处理后的数据
│   ├── train_data.pth # 训练数据文件
│   ├── test_data.pth  # 测试数据文件
│   ├── processed_data.pth # 处理后的数据文件
├── model/             # 模型文件夹，用于存放训练好的模型
│   ├── best_pinn_model_epoch_2000.pth # 最好的模型文件
```

## 环境要求
- Python 3.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm

## 安装依赖
```bash
pip install torch numpy pandas scikit-learn tqdm
```

## 使用说明

### 数据加载
数据加载模块 `data_loader.py` 负责加载和处理数据。主要功能包括：
- 从原始数据文件中读取数据。
- 解析电压和浓度列。
- 对数据进行归一化处理。
- 划分训练集和测试集。
- 保存处理后的数据。

### 模型定义
模型定义文件 `model.py` 中定义了物理信息神经网络（PNN）模型 `PINN`。模型结构包括多个全连接层和激活函数，最终输出 `V` 和 `N`。

### 训练模型
训练模块 `trainer.py` 负责模型的训练。使用 `Trainer` 类进行训练，主要步骤包括：
- 初始化训练器，包括数据加载器、模型、设备和训练参数。
- 进行多轮训练，每轮训练中进行前向传播、计算损失、反向传播和优化。
- 显示训练进度和损失信息。

### 评估模型
评估模块 `evaluate.py` 用于在测试集上评估训练好的模型。主要步骤包括：
- 加载最好的模型。
- 加载测试集数据。
- 在测试集上进行预测。
- 计算评价指标，如均方误差（MSE）、平均绝对误差（MAE）、均方根误差（RMSE）和决定系数（R²）。

## 运行示例
### 训练模型
```python
from trainer import Trainer
from data_loader import DataLoader
from model import PINN

# 初始化数据加载器
data_loader = DataLoader()

# 初始化模型
model = PINN(input_dim=3, hidden_dim=128)

# 初始化训练器
trainer = Trainer(data_loader, model, device='cuda', params={})

# 开始训练
start_epoch = 0
num_epochs = 100
total_epochs = 100
trainer.train(start_epoch, num_epochs, total_epochs)
```

### 评估模型
```python
from evaluate import evaluate_model

evaluate_model()
```

## 注意事项
- 确保数据文件 `data/train_data.pth` 和 `data/test_data.pth` 存在。
- 训练好的模型文件 `model/best_pinn_model_epoch_2000.pth` 用于评估模型。
