import numpy as np

from src.config import SimulationConfig


class DualBrainAgents:
    """
    【修改点】重新包装为 Multi-Agent Actor-Critic (MAAC) 的变体。
    Critic 负责评估价值 (QL 和 QS)，Actor 负责输出策略 (theta 和 Softmax)。
    使用向量化张量同时管理 L x L 个体。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.L = config.L
        self.row_idx, self.col_idx = np.indices((self.L, self.L))

        # 【修改点】Q^L: 6 个局部状态 x 2 个动作
        self.Q_L = np.zeros((self.L, self.L, 6, 2), dtype=float)

        # Q^S: 9 个社会状态 x 2 个动作
        self.Q_S = np.zeros((self.L, self.L, 9, 2), dtype=float)
        # theta -> w = sigmoid(theta) (Actor 策略参数)
        self.theta = np.zeros((self.L, self.L), dtype=float)

        # 策略梯度基线 (Actor 评估基线)
        self.baseline = np.zeros((self.L, self.L), dtype=float)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def get_weights(self) -> np.ndarray:
        return self._sigmoid(self.theta)

    def _get_q_values(self, local_states: np.ndarray, social_states: np.ndarray):
        q_l_vals = self.Q_L[self.row_idx, self.col_idx, local_states]
        q_s_vals = self.Q_S[self.row_idx, self.col_idx, social_states]
        return q_l_vals, q_s_vals

    def choose_action_and_get_grad(self, local_states: np.ndarray, social_states: np.ndarray):
        # Critic 评估价值
        q_l_vals, q_s_vals = self._get_q_values(local_states, social_states)

        # Actor 输出融合策略
        w = self.get_weights()
        w_expanded = w[..., np.newaxis]

        # Q_tot(a; theta) = (1-w) Q^L + w Q^S
        q_tot = (1.0 - w_expanded) * q_l_vals + w_expanded * q_s_vals

        # Softmax
        q_scaled = q_tot / self.config.tau
        q_scaled -= np.max(q_scaled, axis=-1, keepdims=True)
        exp_q = np.exp(q_scaled)
        pi = exp_q / np.sum(exp_q, axis=-1, keepdims=True)

        # 二元动作采样：P(a=1)=pi[...,1]
        rand_matrix = np.random.rand(self.L, self.L)
        actions = (rand_matrix < pi[..., 1]).astype(np.int8)

        # grad_theta log pi(a) = (1/tau) [D(a) - E_pi D]
        # D(a) = dQ_tot(a)/dtheta = w(1-w)(Q^S(a)-Q^L(a))
        D = w_expanded * (1.0 - w_expanded) * (q_s_vals - q_l_vals)
        expected_D = np.sum(pi * D, axis=-1)
        actual_D = D[self.row_idx, self.col_idx, actions]
        grad_log_pi = (actual_D - expected_D) / self.config.tau

        return actions, grad_log_pi

    def update_local_q(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        norm_payoffs: np.ndarray,
        next_states: np.ndarray,
    ):
        q_current = self.Q_L[self.row_idx, self.col_idx, states, actions]
        q_next_max = np.max(self.Q_L[self.row_idx, self.col_idx, next_states], axis=-1)
        td_target = norm_payoffs + self.config.gamma_1 * q_next_max
        td_error = td_target - q_current
        self.Q_L[self.row_idx, self.col_idx, states, actions] += self.config.alpha_1 * td_error

    def update_social_q_and_pg(
        self,
        delayed_states: np.ndarray,
        delayed_actions: np.ndarray,
        G_returns: np.ndarray,
        current_states: np.ndarray,
        delayed_grad_log_pi: np.ndarray,
    ):
        # Q^S 的 N 步目标
        q_current = self.Q_S[self.row_idx, self.col_idx, delayed_states, delayed_actions]
        q_next_max = np.max(self.Q_S[self.row_idx, self.col_idx, current_states], axis=-1)
        gamma_N = self.config.gamma_2 ** self.config.N_steps
        G_target = G_returns + gamma_N * q_next_max

        td_error = G_target - q_current
        self.Q_S[self.row_idx, self.col_idx, delayed_states, delayed_actions] += self.config.alpha_2 * td_error

        # Actor 策略更新：theta <- theta + eta * A * grad log pi
        advantage = G_target - self.baseline
        self.theta += self.config.eta * advantage * delayed_grad_log_pi

        # 平滑更新 baseline
        self.baseline = (
            (1.0 - self.config.baseline_alpha) * self.baseline
            + self.config.baseline_alpha * G_target
        )