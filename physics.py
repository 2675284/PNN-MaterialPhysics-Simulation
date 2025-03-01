import torch

class PhysicsLoss:
    def __init__(self, params):
        self.params = params

    def compute(self, model, x):
        # 确保输入 x 有梯度信息
        x = x.requires_grad_(True)

        # 前向传播
        V_N = model(x)
        V = V_N[:, 0]
        N = V_N[:, 1]

        # 计算 V 关于 x 的梯度
        grad_V = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]

        # 计算 V 的拉普拉斯算子
        grad_V_sum = grad_V.sum(dim=1)  # 对每个样本的梯度求和，确保形状匹配
        laplacian_V = torch.autograd.grad(grad_V_sum, x, grad_outputs=torch.ones_like(grad_V_sum), create_graph=True)[0]

        # 计算物理损失
        physics_loss = torch.mean(laplacian_V ** 2)  # 示例物理损失

        return physics_loss