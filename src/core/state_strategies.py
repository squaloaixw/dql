from abc import ABC, abstractmethod

import numpy as np

from src.config import SimulationConfig


class StateProvider(ABC):
    """状态提取抽象基类。"""

    def __init__(self, config: SimulationConfig):
        self.config = config

    @abstractmethod
    def get_state(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class LocalStateProvider(StateProvider):
    """
    局部经验状态，共 2 x 4 = 8 种状态。

    这里将收益归一化统一改为“基于理论收益范围的绝对归一化”，
    不再使用局部相对 min-max 归一化。
    """

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _theoretical_payoff_bounds(self):
        """
        对当前 SPGG 收益定义，理论收益上下界为：

        payoff_max:
            背叛者位于全合作环境中，
            每个参与组都得到 r*c，共 5 个组：
                payoff_max = 5 * r * c

        payoff_min:
            合作者位于其余人全背叛环境中，
            每个参与组只有自己 1 个合作者，
            总收益为 r*c，成本为 5*c：
                payoff_min = r*c - 5*c = c*(r-5)

        为兼容 r >= 5 的情况，整体最小值取 min(0, c*(r-5)).
        """
        payoff_min = min(0.0, self.config.c * (self.config.r - 5.0))
        payoff_max = 5.0 * self.config.r * self.config.c
        return payoff_min, payoff_max

    def _normalize_payoff(self, payoffs: np.ndarray) -> np.ndarray:
        """
        全局统一采用理论收益范围绝对归一化。
        """
        if self.config.use_z_score:
            # 如果你仍想保留 z-score 选项，可在这里保留；
            # 但若要“完全统一成绝对归一化”，建议不要走这个分支。
            z = (payoffs - np.mean(payoffs)) / (np.std(payoffs) + self.config.xi)
            return self._sigmoid(z)

        payoff_min, payoff_max = self._theoretical_payoff_bounds()
        denom = payoff_max - payoff_min + self.config.xi
        norm_p = (payoffs - payoff_min) / denom

        # 避免精确到 0 或 1，给学习留一点缓冲
        return np.clip(norm_p, self.config.xi, 1.0 - self.config.xi)

    def get_state(self, last_actions: np.ndarray, last_payoffs: np.ndarray) -> np.ndarray:
        norm_p = self._normalize_payoff(last_payoffs)
        payoff_bins = np.digitize(norm_p, bins=[0.25, 0.5, 0.75])
        return last_actions * 4 + payoff_bins


class SocialStateProvider(StateProvider):
    """
    社会观察状态，共 3 x 3 = 9 种状态。
    """

    def get_state(
        self,
        avg_neighbor_coop: np.ndarray,
        avg_neighbor_norm_payoff: np.ndarray,
    ) -> np.ndarray:
        coop_bins = np.digitize(avg_neighbor_coop, bins=[1.0 / 3.0, 2.0 / 3.0])
        payoff_bins = np.digitize(avg_neighbor_norm_payoff, bins=[1.0 / 3.0, 2.0 / 3.0])
        return coop_bins * 3 + payoff_bins