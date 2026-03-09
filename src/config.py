import yaml
import json
from dataclasses import dataclass, field, fields
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """
    基于局部经验与社会观察的双视角融合模型的全局配置类。
    """
    # ==========================================
    # 1. 网络与公共物品博弈基础参数 (Spatial PGG)
    # ==========================================
    L: int = 100  # 网格边长，总节点数 M = L * L (默认 100x100 = 10000)
    r: float = 3.0  # 协同因子 (Synergy factor)
    c: float = 1.0  # 合作成本 (Cost of cooperation)
    iterations: int = 3000  # 最大迭代轮数

    # ==========================================
    # 2. 状态空间参数 (State Space)
    # ==========================================
    N_steps: int = 5  # 社会观察的延迟步数 N (用于计算过去 N 步的平均合作率和累积收益)
    xi: float = 1e-6  # 归一化极小值 (防止除以 0)，如果你采纳审稿意见改为 Z-score，此项可废弃
    use_z_score: bool = True  # 【审稿人建议】是否使用 Z-score 替代原有的 Min-Max 归一化以防除零溢出

    # ==========================================
    # 3. 局部 Q 表参数 (Local Q-Learning - Fast System)
    # ==========================================
    alpha_1: float = 0.1  # 局部 Q 表学习率
    gamma_1: float = 0.9  # 局部 Q 表折扣因子

    # ==========================================
    # 4. 社会 Q 表参数 (Social Q-Learning - Slow System)
    # ==========================================
    alpha_2: float = 0.1  # 社会 Q 表学习率
    gamma_2: float = 0.9  # 社会 Q 表折扣因子

    # ==========================================
    # 5. 策略梯度与动作选择参数 (Policy Gradient & Action Selection)
    # ==========================================
    tau: float = 0.1  # Softmax 温度参数 (越小越贪心，越大越随机，杜绝 epsilon-greedy)
    eta: float = 0.01  # 策略梯度学习率 (用于更新无界参数 theta_i)
    baseline_alpha: float = 0.05  # 基线 b_i(t) 的平滑更新率 (用于降低 PG 方差)

    # ==========================================
    # 6. 工程与辅助参数
    # ==========================================
    seed: int = 42  # 随机种子，保证实验可复现

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """从字典创建配置对象，自动过滤掉字典中多余的、未在 dataclass 中定义的键。"""
        valid_init_keys = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_init_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'SimulationConfig':
        """从 YAML 文件加载配置。"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> 'SimulationConfig':
        """从 JSON 文件加载配置。"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """将当前配置转换为字典，方便保存到 HDF5 或 JSON 以便实验记录。"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}