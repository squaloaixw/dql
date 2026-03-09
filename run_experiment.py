import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 导入我们刚才写好的配置和环境类
from src.config import SimulationConfig
from src.core.spgg_model import SPGGEnvironment


def plot_cooperation_rate(coop_rate_history: list, r_val: float, output_dir: str = "results"):
    """
    绘制合作率随迭代次数变化的折线图，并保存为高精度图片。

    Args:
        coop_rate_history: 合作率历史列表
        r_val: 当前实验的协同因子 r (用于图表标题和命名)
        output_dir: 图片保存路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置学术绘图风格 (可选，让图表更美观)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制折线
    ax.plot(coop_rate_history, color='#d62728', linewidth=2.5, label=f'Dual-Brain Model (r={r_val})')

    # 设置标题和标签
    ax.set_title(f'Evolution of Cooperation Rate (Synergy Factor r={r_val})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Fraction of Cooperators', fontsize=12)

    # 设置 Y 轴范围为 0 到 1 (代表 0% 到 100% 合作率)
    ax.set_ylim(-0.05, 1.05)

    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=11)

    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'coop_evolution_r{r_val}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 合作率演化折线图已生成并保存至: {save_path}")


def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="运行基于局部经验与社会观察的双视角融合 SPGG 模型")
    parser.add_argument('--r', type=float, default=3.0, help='公共物品博弈的协同因子 r (Synergy factor)')
    parser.add_argument('--L', type=int, default=100, help='网格边长 L (默认 100)')
    parser.add_argument('--iterations', type=int, default=10000, help='最大迭代轮数 (默认 10000)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认 42)')
    args = parser.parse_args()

    print("=" * 50)
    print(f"初始化实验配置: L={args.L}, r={args.r}, Iterations={args.iterations}, Seed={args.seed}")
    print("=" * 50)


    # 2. 实例化配置类
    config = SimulationConfig(
        L=args.L,
        r=args.r,
        iterations=args.iterations,
        seed=args.seed
    )

    # 3. 初始化 SPGG 核心环境
    env = SPGGEnvironment(config)

    # 4. 运行模拟，获取历史数据
    coop_rate_history = env.run()

    # 5. 调用绘图函数
    plot_cooperation_rate(coop_rate_history, config.r)


if __name__ == "__main__":
    main()