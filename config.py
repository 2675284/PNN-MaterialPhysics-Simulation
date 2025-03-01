# config.py - 物理常数与材料参数
class MaterialParameters:
    def __init__(self):
        # 基本物理常数
        self.q = 1.602176634e-19      # 电子电荷量 (C)
        self.kB = 1.380649e-23        # 玻尔兹曼常数 (J/K)
        self.epsilon0 = 8.8541878e-12 # 真空介电常数 (F/m)

        # 硅材料参数
        self.epsilon_r = 11.7          # 相对介电常数
        self.mu_n = 0.14              # 电子迁移率 (m²/(V·s))
        self.mu_p = 0.045             # 空穴迁移率 (m²/(V·s))

        # 掺杂浓度（转换为m^-3）
        self.Na = 1e17 * 1e6          # P区掺杂：1e17 cm⁻³ → 1e23 m⁻³
        self.Nd = 1e15 * 1e6          # N区掺杂：1e15 cm⁻³ → 1e21 m⁻³

        # 计算扩散系数（爱因斯坦关系）
        self.T = 300                   # 温度 (K)
        self.Vt = (self.kB * self.T) / self.q  # 热电压 (≈0.0259 V)
        self.D_n = self.mu_n * self.Vt  # 电子扩散系数 (m²/s)
        self.D_p = self.mu_p * self.Vt  # 空穴扩散系数 (m²/s)
