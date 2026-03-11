from collections import deque

import numpy as np
from scipy.ndimage import convolve

from src.config import SimulationConfig
from src.core.q_pg_agent import DualBrainAgents
from src.core.state_strategies import LocalStateProvider, SocialStateProvider


class SPGGEnvironment:
    """
    空间公共物品博弈环境。
    关键修正：
    1. 局部收益归一化改为与模型一致的“邻域相对归一化”；
    2. 合作率记录改为记录“执行当前轮动作后的状态”，避免日志滞后一轮；
    3. 历史数据入队时显式 copy，避免后续维护时出现引用语义歧义。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.L = config.L
        self.N = config.N_steps

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.local_provider = LocalStateProvider(config)
        self.social_provider = SocialStateProvider(config)
        self.agent = DualBrainAgents(config)

        self.kernel_5 = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        self.kernel_4 = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )

        # 当前“最近一轮”的动作与收益，用来构造下一轮状态
        self.actions = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        self.payoffs = self._calculate_payoffs(self.actions)

        # 存过去 N 步的动作 / 归一化收益，用于社会观察状态
        self.recent_actions = deque(maxlen=self.N)
        self.recent_norm_payoffs = deque(maxlen=self.N)

        # 存 (s^S(t), a(t), \tilde{pi}(t), grad_log_pi(t))，供 N 步延迟更新
        self.history_buffer = deque(maxlen=self.N)

        self.coop_rate_history = []

    def _calculate_payoffs(self, actions: np.ndarray) -> np.ndarray:
        """
        标准 SPGG：
        - 每个节点参与 5 个五人小组（自己为中心 + 四个邻居为中心）；
        - 每次合作在参与的小组中支付成本 c；
        - 小组总投入乘以 r 后由 5 名成员均分。
        """
        Nc = convolve(actions.astype(float), self.kernel_5, mode="wrap")
        group_benefit = (self.config.r * self.config.c * Nc) / 5.0
        total_benefit = convolve(group_benefit, self.kernel_5, mode="wrap")
        total_payoffs = total_benefit - 5.0 * self.config.c * actions
        return total_payoffs

    def _get_social_state(self) -> np.ndarray:
        """
        严格使用“过去 N 步”的邻居平均合作率与邻居平均归一化收益。
        队列里保存的就是当前决策前可观测到的最近 N 个历史截面。
        """
        if len(self.recent_actions) == 0:
            avg_action = self.actions.astype(float)
        else:
            avg_action = np.mean(np.stack(self.recent_actions, axis=0), axis=0)

        if len(self.recent_norm_payoffs) == 0:
            avg_norm_payoff = self.local_provider._normalize_payoff(self.payoffs)
        else:
            avg_norm_payoff = np.mean(np.stack(self.recent_norm_payoffs, axis=0), axis=0)

        neighbor_avg_coop = convolve(avg_action, self.kernel_4, mode="wrap") / 4.0
        neighbor_avg_norm_payoff = convolve(avg_norm_payoff, self.kernel_4, mode="wrap") / 4.0

        return self.social_provider.get_state(neighbor_avg_coop, neighbor_avg_norm_payoff)

    def run(self):
        print(f"🚀 启动双视角融合 SPGG 模拟 (L={self.L}, iterations={self.config.iterations})...")

        for t in range(self.config.iterations):
            # 当前可观测历史（对应状态 s(t) 的输入）
            norm_payoffs = self.local_provider._normalize_payoff(self.payoffs)
            self.recent_actions.append(self.actions.copy())
            self.recent_norm_payoffs.append(norm_payoffs.copy())

            # 1. 提取当前状态
            s_L = self.local_provider.get_state(self.actions, self.payoffs)
            s_S = self._get_social_state()

            # 2. 满 N 步后执行 Q^S + PG 延迟更新
            if len(self.history_buffer) == self.N:
                G_returns = np.zeros((self.L, self.L), dtype=float)
                for m, (_, _, reward_m, _) in enumerate(self.history_buffer):
                    G_returns += (self.config.gamma_2 ** m) * reward_m

                delayed_s_S, delayed_a, _, delayed_grad = self.history_buffer[0]
                self.agent.update_social_q_and_pg(
                    delayed_states=delayed_s_S,
                    delayed_actions=delayed_a,
                    G_returns=G_returns,
                    current_states=s_S,
                    delayed_grad_log_pi=delayed_grad,
                )

            # 3. 根据融合策略选动作
            new_actions, grad_log_pi = self.agent.choose_action_and_get_grad(s_L, s_S)

            # 4. 环境流转并得到收益
            new_payoffs = self._calculate_payoffs(new_actions)
            new_norm_payoffs = self.local_provider._normalize_payoff(new_payoffs)

            # 5. 单步局部 Q 更新
            s_L_next = self.local_provider.get_state(new_actions, new_payoffs)
            self.agent.update_local_q(
                states=s_L,
                actions=new_actions,
                norm_payoffs=new_norm_payoffs,
                next_states=s_L_next,
            )

            # 6. 将本轮经验压入历史缓冲，用于未来 N 步更新
            self.history_buffer.append(
                (
                    s_S.copy(),
                    new_actions.copy(),
                    new_norm_payoffs.copy(),
                    grad_log_pi.copy(),
                )
            )

            # 7. 推进到下一轮
            self.actions = new_actions
            self.payoffs = new_payoffs

            # 记录“当前轮执行后的”合作率，避免日志相位滞后
            coop_rate = float(np.mean(self.actions))
            self.coop_rate_history.append(coop_rate)

            if (t + 1) % self.config.log_interval == 0:
                w = self.agent.get_weights()
                theta = self.agent.theta

                mean_w = float(np.mean(w))
                std_w = float(np.std(w))
                w_p10, w_p50, w_p90 = np.percentile(w, [10, 50, 90])

                mean_theta = float(np.mean(theta))
                std_theta = float(np.std(theta))
                mean_abs_theta = float(np.mean(np.abs(theta)))

                mean_abs_grad = float(np.mean(np.abs(grad_log_pi)))

                w_c = float(np.mean(w[self.actions == 1])) if np.any(self.actions == 1) else np.nan
                w_d = float(np.mean(w[self.actions == 0])) if np.any(self.actions == 0) else np.nan

                print(
                    f"Iteration {t + 1:05d}/{self.config.iterations} | "
                    f"Coop Rate: {coop_rate:.4f} | "
                    f"mean w: {mean_w:.6f} | std w: {std_w:.6f} | "
                    f"w[p10,p50,p90]=({w_p10:.6f}, {w_p50:.6f}, {w_p90:.6f}) | "
                    f"mean theta: {mean_theta:.6e} | std theta: {std_theta:.6e} | "
                    f"mean|theta|: {mean_abs_theta:.6e} | "
                    f"mean|grad|: {mean_abs_grad:.6e}"
                    f"... | mean w(C): {w_c:.6f} | mean w(D): {w_d:.6f}"
                )

        print("✅ 模拟结束！")
        return self.coop_rate_history