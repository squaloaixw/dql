import argparse
import os

import matplotlib.pyplot as plt

from src.config import SimulationConfig
from src.core.spgg_model import SPGGEnvironment


def plot_cooperation_rate(coop_rate_history: list, r_val: float, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coop_rate_history, linewidth=2.0, label=f"Dual-Brain Model (r={r_val})")
    ax.set_title(f"Evolution of Cooperation Rate (r={r_val})", fontsize=14)
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Fraction of Cooperators", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"coop_evolution_r{r_val}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📊 合作率演化折线图已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="运行双视角融合 SPGG 模型")
    parser.add_argument("--r", type=float, default=3.6, help="协同因子 r")
    parser.add_argument("--L", type=int, default=100, help="网格边长 L")
    parser.add_argument("--eta", type=float, default=0.05, help="学习率")
    parser.add_argument("--iterations", type=int, default=10000, help="最大迭代轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 50)
    print(
        f"初始化实验配置: L={args.L}, r={args.r}, "
        f"Iterations={args.iterations}, Seed={args.seed}"
    )
    print("=" * 50)

    config = SimulationConfig(
        L=args.L,
        r=args.r,
        iterations=args.iterations,
        seed=args.seed,
    )

    env = SPGGEnvironment(config)
    coop_rate_history = env.run()
    plot_cooperation_rate(coop_rate_history, config.r)


if __name__ == "__main__":
    main()