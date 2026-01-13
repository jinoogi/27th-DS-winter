"""[과제 1 - Part 1] 10-armed Bandit 템플릿

요구사항(요약)
- k=10
- q*(a) ~ N(0,1) (run마다 새로 샘플, 이후 고정)
- R_t ~ N(q*(A_t), 1)
- time steps=2000, runs=500
- 알고리즘 3종:
  1) epsilon-greedy (epsilon=0.1), sample-average 업데이트
  2) optimistic init: Q_1(a)=5.0, epsilon=0, alpha=0.1 고정 step-size
  3) UCB: c=2
- 결과: steps별 평균 보상 그래프(3개를 1 plot에)

주의: 핵심 로직은 TODO로 비워두었습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import time


AlgoName = Literal["eps_greedy", "optimistic", "ucb"]


@dataclass
class BanditSpec:
    k: int = 10
    reward_noise_std: float = 1.0


class NormalKArmedBandit:
    """Stationary k-armed bandit with Gaussian rewards."""

    def __init__(self, spec: BanditSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=(self.spec.k,))

    def step(self, action: int) -> float:
        mean = float(self.q_star[action])
        return float(self.rng.normal(loc=mean, scale=self.spec.reward_noise_std))


@dataclass
class AgentState:
    q_est: np.ndarray  # estimated action values Q_t(a)
    n: np.ndarray  # action counts N_t(a)


def select_action_eps_greedy(state: AgentState, rng: np.random.Generator, epsilon: float) -> int:
    """TODO: epsilon-greedy action selection."""
    raise NotImplementedError


def update_sample_average(state: AgentState, action: int, reward: float) -> None:
    """TODO: sample-average update for Q."""
    raise NotImplementedError


def select_action_ucb(state: AgentState, t: int, c: float) -> int:
    """TODO: UCB action selection."""
    raise NotImplementedError


def update_constant_step_size(state: AgentState, action: int, reward: float, alpha: float) -> None:
    """TODO: constant step-size update."""
    raise NotImplementedError


def run_one(algo: AlgoName, *, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Run one bandit instance for `steps` steps.

    Returns:
      rewards: shape (steps,)
    """
    bandit = NormalKArmedBandit(BanditSpec(k=10), rng=rng)

    q0 = np.zeros((bandit.spec.k,), dtype=float)
    if algo == "optimistic":
        q0 = np.full((bandit.spec.k,), 5.0, dtype=float)

    state = AgentState(q_est=q0.copy(), n=np.zeros((bandit.spec.k,), dtype=int))

    rewards = np.zeros((steps,), dtype=float)

    eps = 0.1
    alpha = 0.1
    c = 2.0

    for t in range(1, steps + 1):
        if algo == "eps_greedy":
            action = select_action_eps_greedy(state, rng=rng, epsilon=eps)
        elif algo == "optimistic":
            action = select_action_eps_greedy(state, rng=rng, epsilon=0.0)
        elif algo == "ucb":
            action = select_action_ucb(state, t=t, c=c)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        reward = bandit.step(action)
        rewards[t - 1] = reward

        if algo == "eps_greedy":
            update_sample_average(state, action=action, reward=reward)
        elif algo == "optimistic":
            update_constant_step_size(state, action=action, reward=reward, alpha=alpha)
        elif algo == "ucb":
            update_sample_average(state, action=action, reward=reward)

    return rewards


def run_experiment(
    *,
    runs: int = 500,
    steps: int = 2000,
    seed: int = 0,
    progress: bool = True,
    progress_every: int = 10,
) -> dict[str, np.ndarray]:
    """Run all algorithms and average over runs.

    progress: runs 진행 로그 출력 여부
    progress_every: 몇 run마다 로그를 찍을지
    """
    rng_master = np.random.default_rng(seed)
    algos: list[AlgoName] = ["eps_greedy", "optimistic", "ucb"]

    avg_rewards: dict[str, np.ndarray] = {}

    for algo in algos:
        rewards_runs = np.zeros((runs, steps), dtype=float)

        t0 = time.perf_counter()

        for r in range(runs):
            # separate RNG per run for reproducibility
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            rewards = run_one(algo, steps=steps, rng=rng)
            rewards_runs[r] = rewards

            if progress:
                done = r + 1
                should_print = (done == 1) or (done == runs) or (progress_every > 0 and done % progress_every == 0)
                if should_print:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (runs - done) / rate if rate > 0 else float("inf")
                    msg = f"[{algo}] {done}/{runs} ({done / runs * 100:5.1f}%)  elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s"
                    end = "\n" if done == runs else "\r"
                    print(msg, end=end, flush=True)

        avg_rewards[algo] = rewards_runs.mean(axis=0)

    return {
        "avg_rewards_eps_greedy": avg_rewards["eps_greedy"],
        "avg_rewards_optimistic": avg_rewards["optimistic"],
        "avg_rewards_ucb": avg_rewards["ucb"],
    }


def plot_results(results: dict[str, np.ndarray], *, title: str = "10-armed Bandit") -> None:
    x = np.arange(1, results["avg_rewards_eps_greedy"].shape[0] + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, results["avg_rewards_eps_greedy"], label="epsilon-greedy (e=0.1)")
    plt.plot(x, results["avg_rewards_optimistic"], label="optimistic init (Q1=5, a=0.1)")
    plt.plot(x, results["avg_rewards_ucb"], label="UCB (c=2)")

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run() -> None:
    results = run_experiment(runs=500, steps=2000, seed=0)
    plot_results(results)


if __name__ == "__main__":
    run()
