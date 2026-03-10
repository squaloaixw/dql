import numpy as np
from abc import ABC, abstractmethod
from scipy.ndimage import maximum_filter, minimum_filter
from src.config import SimulationConfig


class StateProvider(ABC):
    """
    状态提取的抽象基类。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # 冯·诺依曼邻域（上下左右 + 自身）的卷积核/足迹 [cite: 4, 6]
        self.von_neumann_footprint = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

    @abstractmethod
    def get_state(self, *args, **kwargs) -> np.ndarray:
        pass


class LocalStateProvider(StateProvider):
    """
    计算局部经验状态 (Local Experience State, s^L_i(t))
    包含 8 种状态：2 种动作 x 4 种收益分箱 [cite: 11, 13]。
    """

    def _normalize_payoff(self, payoffs: np.ndarray) -> np.ndarray:
        """
        计算归一化收益：使用 SPGG 的理论上下界进行静态映射，
        保证强化学习奖励函数的绝对平稳性（Stationarity）。
        """
        # 计算理论收益上下界 (基于 k=4 的冯·诺依曼邻域，每人参与 5 场博弈)
        max_p = 5.0 * self.config.r * self.config.c
        min_p = -5.0 * self.config.c

        # 线性映射到 [0, 1]
        norm_p = (payoffs - min_p) / (max_p - min_p)

        # 裁剪以防浮点误差越界
        return np.clip(norm_p, 0.0, 1.0)

    def get_state(self, last_actions: np.ndarray, last_payoffs: np.ndarray) -> np.ndarray:
        """
        获取局部经验状态矩阵 (LxL)。

        Args:
            last_actions: t-1 时刻的动作矩阵，0=背叛，1=合作 [cite: 7, 11]
            last_payoffs: t-1 时刻的真实收益矩阵 [cite: 11]

        Returns:
            LxL 的整数矩阵，取值范围 [0, 7]
        """
        # 1. 相对收益归一化 [cite: 12]
        norm_p = self._normalize_payoff(last_payoffs)

        # 2. Bin_4 离散化 (0, 1, 2, 3) [cite: 11]
        # bins=[0.25, 0.5, 0.75] 会将数据分为: <0.25, 0.25-0.5, 0.5-0.75, >0.75
        payoff_bins = np.digitize(norm_p, bins=[0.25, 0.5, 0.75])

        # 3. 组合状态: action * 4 + payoff_bin
        # 若 action=0, states: 0, 1, 2, 3
        # 若 action=1, states: 4, 5, 6, 7
        local_states = last_actions * 4 + payoff_bins
        return local_states


class SocialStateProvider(StateProvider):
    """
    计算社会观察状态 (Social Observation State, s^S_i(t))
    包含 9 种状态：3 种合作率分箱 x 3 种收益分箱 [cite: 15, 18]。
    """

    def get_state(self, avg_neighbor_coop: np.ndarray, avg_neighbor_norm_payoff: np.ndarray) -> np.ndarray:
        """
        获取社会观察状态矩阵 (LxL)。
        注意：传入的参数已经是过去 N 步的“邻居平均值”，这部分累加逻辑由主循环维护传入 [cite: 15, 16, 17]。

        Args:
            avg_neighbor_coop: 过去 N 步的邻居平均合作率，范围 [0, 1] [cite: 16]
            avg_neighbor_norm_payoff: 过去 N 步的邻居平均归一化收益，范围约 [0, 1] [cite: 17]

        Returns:
            LxL 的整数矩阵，取值范围 [0, 8] [cite: 18]
        """
        # 1. Bin_3 合作率离散化 (0, 1, 2)
        # bins=[0.333, 0.666]
        coop_bins = np.digitize(avg_neighbor_coop, bins=[0.333, 0.666])

        # 2. Bin_3 归一化收益离散化 (0, 1, 2)
        payoff_bins = np.digitize(avg_neighbor_norm_payoff, bins=[0.333, 0.666])

        # 3. 组合状态: coop_bin * 3 + payoff_bin [cite: 18]
        # 取值: 0, 1, 2, 3, 4, 5, 6, 7, 8
        social_states = coop_bins * 3 + payoff_bins
        return social_states