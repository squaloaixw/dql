import json
from dataclasses import dataclass, fields
from typing import Any, Dict

import yaml


@dataclass
class SimulationConfig:
    """
    基于局部经验与社会观察的双视角 Q-Learning + 策略梯度融合模型配置。
    """

    # 1. Spatial PGG 基础参数
    L: int = 100
    r: float = 3.8
    c: float = 1.0
    iterations: int = 10000

    # 2. 状态空间 / 回报参数
    N_steps: int = 5
    xi: float = 1e-6
    use_z_score: bool = False

    # 3. 局部 Q 表
    alpha_1: float = 0.5
    gamma_1: float = 0.5

    # 4. 社会 Q 表
    alpha_2: float = 0.1
    gamma_2: float = 0.9

    # 5. 动作选择与策略梯度
    tau: float = 0.1
    eta: float = 0.5
    baseline_alpha: float = 0.05

    # 6. 工程参数
    seed: int = 42
    log_interval: int = 5000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        valid_init_keys = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_init_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> "SimulationConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}