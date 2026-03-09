import numpy as np
from collections import deque
from scipy.ndimage import convolve

from src.config import SimulationConfig
from src.core.state_strategies import LocalStateProvider, SocialStateProvider
from src.core.q_pg_agent import DualBrainAgents


class SPGGEnvironment:
    """
    空间公共物品博弈环境核心模块，整合双大脑智能体与状态提取器。
    使用完全向量化的方式计算群体博弈收益与状态流转。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.L = config.L
        self.N = config.N_steps

        # 设定随机种子
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # 1. 实例化策略与大脑
        self.local_provider = LocalStateProvider(config)
        self.social_provider = SocialStateProvider(config)
        self.agent = DualBrainAgents(config)

        # 2. 卷积核定义
        # 用于计算自身参与的 5 人小组 (中心 + 4 个邻居)
        self.kernel_5 = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        # 用于仅计算 4 个邻居 (不含自身，用于社会观察的邻居均值)
        self.kernel_4 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        # 3. 环境状态初始化 (随机初始化 0 或 1)
        self.actions = np.random.randint(0, 2, size=(self.L, self.L))
        self.payoffs = self._calculate_payoffs(self.actions)

        # 4. 历史缓冲队列 (用于 N 步延迟更新与平均计算)
        self.recent_actions = deque(maxlen=self.N)
        self.recent_norm_payoffs = deque(maxlen=self.N)
        # history_buffer 存放元组: (s^S_t, 动作, 归一化收益, 梯度因子)
        self.history_buffer = deque(maxlen=self.N)

        # 数据追踪
        self.coop_rate_history = []

    def _calculate_payoffs(self, actions: np.ndarray) -> np.ndarray:
        """
        计算公共物品博弈 (PGG) 中每个节点的最终收益。
        严格按照标准 SPGG 规则：每个节点参与 5 场博弈，每次合作付出成本 c。
        """
        # 1. 统计每个 5 人小组内的合作者数量 Nc
        Nc = convolve(actions, self.kernel_5, mode='wrap')

        # 2. 该小组创造的公共收益（被 5 个成员平分后的单份收益）
        group_benefit = (self.config.r * self.config.c * Nc) / 5.0

        # 3. 每个人从自己参与的 5 个小组中获得的收益总和
        total_benefit = convolve(group_benefit, self.kernel_5, mode='wrap')

        # 4. 减去自身作为合作者在 5 个小组中付出的总成本 (5 * c)
        total_payoffs = total_benefit - 5.0 * self.config.c * actions
        return total_payoffs

    def _get_social_state(self) -> np.ndarray:
        """
        提取社会观察状态 s^S(t)。需要计算过去 N 步的邻居平均特征。
        """
        # 取过去 N 步的时间维度的均值
        avg_action = np.mean(self.recent_actions, axis=0) if self.recent_actions else self.actions
        avg_norm_payoff = np.mean(self.recent_norm_payoffs,
                                  axis=0) if self.recent_norm_payoffs else self.local_provider._normalize_payoff(
            self.payoffs)

        # 计算空间维度上的邻居均值 (除以 4 个邻居)
        neighbor_avg_coop = convolve(avg_action, self.kernel_4, mode='wrap') / 4.0
        neighbor_avg_norm_payoff = convolve(avg_norm_payoff, self.kernel_4, mode='wrap') / 4.0

        return self.social_provider.get_state(neighbor_avg_coop, neighbor_avg_norm_payoff)

    def run(self):
        """
        环境主循环。
        """
        print(f"🚀 启动双视角融合 SPGG 模拟 (L={self.L}, iterations={self.config.iterations})...")

        for t in range(self.config.iterations):
            # 记录合作率指标
            coop_rate = np.mean(self.actions)
            self.coop_rate_history.append(coop_rate)

            # 归一化当前时刻的收益 (为社会状态队列和局部状态提供输入)
            norm_payoffs = self.local_provider._normalize_payoff(self.payoffs)

            # 更新用于计算社会状态的近期历史
            self.recent_actions.append(self.actions)
            self.recent_norm_payoffs.append(norm_payoffs)

            # ==========================================
            # STEP 1: 提取当前状态
            # ==========================================
            s_L = self.local_provider.get_state(self.actions, self.payoffs)
            s_S = self._get_social_state()

            # ==========================================
            # STEP 2: N 步延迟更新 (如果缓冲区满 N 步)
            # ==========================================
            if len(self.history_buffer) == self.N:
                # 计算 N 步累积折扣回报 G_{t-N:t}
                G = np.zeros((self.L, self.L))
                for m, item in enumerate(self.history_buffer):
                    reward_m = item[2]  # buffer 第 2 项是 norm_payoffs (收益)
                    G += (self.config.gamma_2 ** m) * reward_m

                # 取出 t-N 时刻的缓存数据
                delayed_s_S, delayed_a, _, delayed_grad = self.history_buffer[0]

                # 执行社会 Q 表和策略梯度的更新 (此时的目标状态即为当前的 s_S)
                self.agent.update_social_q_and_pg(
                    delayed_states=delayed_s_S,
                    delayed_actions=delayed_a,
                    G_returns=G,
                    current_states=s_S,
                    delayed_grad_log_pi=delayed_grad
                )

            # ==========================================
            # STEP 3: 决策与行动 (前向传播)
            # ==========================================
            new_actions, grad_log_pi = self.agent.choose_action_and_get_grad(s_L, s_S)

            # ==========================================
            # STEP 4: 环境流转，计算新收益
            # ==========================================
            new_payoffs = self._calculate_payoffs(new_actions)
            new_norm_payoffs = self.local_provider._normalize_payoff(new_payoffs)

            # ==========================================
            # STEP 5: 单步即时更新 (局部 Q 表)
            # ==========================================
            # 为了进行 TD 更新，我们需要拿到执行新动作后进入的下一个局部状态 s^L_{t+1}
            s_L_next = self.local_provider.get_state(new_actions, new_payoffs)
            self.agent.update_local_q(s_L, new_actions, new_norm_payoffs, s_L_next)

            # ==========================================
            # STEP 6: 压入历史队列并推进时间
            # ==========================================
            # 注意：存入的是进行决策时的社会状态 s_S、作出的动作、获得的归一化收益以及梯度
            self.history_buffer.append((s_S, new_actions, new_norm_payoffs, grad_log_pi))

            self.actions = new_actions
            self.payoffs = new_payoffs

            # 打印进度
            if (t + 1) % 1000 == 0:
                print(
                    f"Iteration {t + 1:05d}/{self.config.iterations} | Coop Rate: {coop_rate:.4f} | 权重均值 w: {np.mean(1 / (1 + np.exp(-self.agent.theta))):.3f}")

        print("✅ 模拟结束！")
        return self.coop_rate_history