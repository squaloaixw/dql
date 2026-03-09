import numpy as np
from src.config import SimulationConfig


class DualBrainAgents:
    """
    双视角 Q-Learning 与策略梯度融合大脑。
    使用完全向量化的 Numpy 张量运算，同时管理 LxL 个智能体。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.L = config.L

        # 为了高效查询 Q 表，预先生成网格的行索引和列索引
        self.row_idx, self.col_idx = np.indices((self.L, self.L))

        # ==========================================
        # 1. 价值函数 (Critic) - 状态-动作价值表
        # ==========================================
        # 局部 Q 表 (L, L, 8种状态, 2种动作)
        # 初始化为微小的随机数，打破初始对称性
        self.Q_L = np.random.uniform(-0.01, 0.01, size=(self.L, self.L, 8, 2))

        # 社会 Q 表 (L, L, 9种状态, 2种动作)
        self.Q_S = np.random.uniform(-0.01, 0.01, size=(self.L, self.L, 9, 2))

        # ==========================================
        # 2. 策略参数 (Actor) - 个体理性与社会信仰的博弈
        # ==========================================
        # 内部无界参数 theta_i，初始化为 0 (此时 w_i = 0.5，绝对公平)
        self.theta = np.zeros((self.L, self.L))

        # PG 的基线 (Baseline) b_i(t)，用于降低策略梯度的方差，初始化为 0
        self.baseline = np.zeros((self.L, self.L))

    def _get_q_values(self, local_states: np.ndarray, social_states: np.ndarray):
        """内部辅助函数：获取当前状态下的局部和全局 Q 值 (L, L, 2)"""
        q_l_vals = self.Q_L[self.row_idx, self.col_idx, local_states]
        q_s_vals = self.Q_S[self.row_idx, self.col_idx, social_states]
        return q_l_vals, q_s_vals

    def choose_action_and_get_grad(self, local_states: np.ndarray, social_states: np.ndarray):
        """
        前向传播：融合 Q 值 -> Softmax 采样动作 -> 计算策略梯度因子。
        返回动作矩阵，以及为了后续 PG 更新所需要缓存的梯度因子。
        """
        # 1. 获取两套 Q 值
        q_l_vals, q_s_vals = self._get_q_values(local_states, social_states)

        # 2. 计算自适应权重 w_i = sigmoid(theta_i)
        # theta 形状为 (L,L), 增加一个维度变为 (L,L,1) 以便与 (L,L,2) 的 Q 值广播相乘
        w = 1.0 / (1.0 + np.exp(-self.theta))
        w_expanded = w[..., np.newaxis]

        # 3. 融合 Q 值: Q_tot = (1-w) * Q_L + w * Q_S
        Q_tot = (1.0 - w_expanded) * q_l_vals + w_expanded * q_s_vals

        # 4. Softmax 策略分布
        # 减去 max 以防止 exp() 数值溢出 (标准的 Softmax 稳定化技巧)
        Q_tot_scaled = Q_tot / self.config.tau
        Q_max = np.max(Q_tot_scaled, axis=-1, keepdims=True)
        exp_Q = np.exp(Q_tot_scaled - Q_max)
        pi = exp_Q / np.sum(exp_Q, axis=-1, keepdims=True)  # (L, L, 2)

        # 5. 采样动作 (由于是二维选择 0或1，直接用随机数与 pi[..., 1] 比较即可)
        rand_matrix = np.random.rand(self.L, self.L)
        actions = (rand_matrix < pi[..., 1]).astype(int)

        # 6. 计算理论梯度因子 (用于存储到经验回放池，N 步后使用)
        # D_i(a) = w_i * (1 - w_i) * (Q^S(a) - Q^L(a))
        D = w_expanded * (1.0 - w_expanded) * (q_s_vals - q_l_vals)  # (L, L, 2)

        # 期望梯度 sum(pi * D)
        expected_D = np.sum(pi * D, axis=-1, keepdims=True)  # (L, L, 1)

        # 实际采取动作的 D 值
        # actions 形状为 (L,L)，我们提取出对应 action 的 D 值
        actual_D = D[self.row_idx, self.col_idx, actions]  # (L, L)

        # grad_log_pi = (1 / tau) * (D(a) - expected_D)
        grad_log_pi = (actual_D - expected_D.squeeze(-1)) / self.config.tau  # (L, L)

        return actions, grad_log_pi

    def update_local_q(self, states: np.ndarray, actions: np.ndarray, norm_payoffs: np.ndarray,
                       next_states: np.ndarray):
        """
        单步即时更新：局部经验 Q 表 ($Q^L$)
        """
        # 获取当前 Q 值 Q(s, a)
        q_current = self.Q_L[self.row_idx, self.col_idx, states, actions]

        # 获取下一状态的最大 Q 值 max Q(s', a')
        q_next_max = np.max(self.Q_L[self.row_idx, self.col_idx, next_states], axis=-1)

        # TD 目标与误差
        td_target = norm_payoffs + self.config.gamma_1 * q_next_max
        td_error = td_target - q_current

        # 更新 Q_L
        self.Q_L[self.row_idx, self.col_idx, states, actions] += self.config.alpha_1 * td_error

    def update_social_q_and_pg(self, delayed_states: np.ndarray, delayed_actions: np.ndarray,
                               G_returns: np.ndarray, current_states: np.ndarray,
                               delayed_grad_log_pi: np.ndarray):
        """
        N步延迟更新：社会观察 Q 表 ($Q^S$) 以及 策略梯度参数 ($\theta$)
        注意：这修复了原理论中的时序错误，我们用 t-N 时刻的缓存数据进行更新。
        """
        # ==========================================
        # 1. 更新社会 Q 表 (Q^S)
        # ==========================================
        q_current = self.Q_S[self.row_idx, self.col_idx, delayed_states, delayed_actions]
        q_next_max = np.max(self.Q_S[self.row_idx, self.col_idx, current_states], axis=-1)

        # TD Target 为累积真实奖励 G_returns + gamma^N * max Q(s', a')
        gamma_N = self.config.gamma_2 ** self.config.N_steps
        td_target = G_returns + gamma_N * q_next_max
        td_error = td_target - q_current

        self.Q_S[self.row_idx, self.col_idx, delayed_states, delayed_actions] += self.config.alpha_2 * td_error

        # ==========================================
        # 2. 策略梯度更新内部参数 theta
        # ==========================================
        # 计算优势函数 Advantage: A = G - b
        advantage = G_returns - self.baseline

        # 更新 theta: theta <- theta + eta * A * grad_log_pi
        # 使用 delayed_grad_log_pi 因为这是在采取对应 action 时缓存的梯度
        self.theta += self.config.eta * advantage * delayed_grad_log_pi

        # ==========================================
        # 3. 平滑更新 PG 的 Baseline
        # ==========================================
        self.baseline = (1.0 - self.config.baseline_alpha) * self.baseline + \
                        self.config.baseline_alpha * G_returns