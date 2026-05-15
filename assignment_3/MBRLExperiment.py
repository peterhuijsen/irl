#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import time
from typing import Any, Callable, List, Tuple

from matplotlib import pyplot as plt
import numpy as np

from joblib import Parallel, delayed

from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from MBRLEnvironment import WindyGridworld
from scipy.signal import savgol_filter

global_seed = 42

class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')    
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, mean, deviation=None, label=None, x=None):
        if x is None:
            x = np.arange(len(mean))

        if deviation is not None:
            upper_bound = mean + deviation
            lower_bound = mean - deviation
            self.ax.fill_between(x, lower_bound, upper_bound, alpha=0.1, label=f"_hidden_{label}_deviation")

        if label is not None:
            self.ax.plot(x, mean, label=label)
            self.ax.legend()
        else:
            self.ax.plot(x, mean)

    def save(self,name='test.png'):
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly, mode='nearest')

def run_single_repetition(
    spawn_env: Callable[[], WindyGridworld],
    spawn_agent: Callable[[WindyGridworld], Any],
    n_timesteps: int,
    eval_interval: int,
    epsilon: float,
    n_planning_updates: int,
    max_episode_length: int = 100,
    seed: int = global_seed,
) -> np.ndarray:
    np.random.seed(seed)
    env = spawn_env()
    eval_env = spawn_env()
    agent = spawn_agent(env)

    returns = []
    s = env.reset()
    for t in range(n_timesteps):
        if t % eval_interval == 0:
            returns.append(agent.evaluate(eval_env, max_episode_length=max_episode_length))

        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        agent.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

        if done:
            s = env.reset()
        else:
            s = s_next

    return np.array(returns)

def run_multiple_repetitions(
    spawn_env: Callable[[], WindyGridworld],
    spawn_agent: Callable[[WindyGridworld], Any],
    n_timesteps: int,
    eval_interval: int,
    epsilon: float,
    n_planning_updates: int,
    num_repetitions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    results = Parallel(n_jobs=-1)(
        delayed(run_single_repetition)(
            spawn_env, spawn_agent, n_timesteps, eval_interval, epsilon, n_planning_updates,
            seed=global_seed + i,
        )
        for i in range(num_repetitions)
    )
    rewards = np.array(results)
    smoothed = np.array([smooth(rep, window=5) for rep in rewards])
    return np.mean(smoothed, axis=0), np.std(smoothed, axis=0)

def run_experiment(
    spawn_env: Callable[[], WindyGridworld],
    spawn_agent: Callable[[WindyGridworld], Any],
    parameters: List[int],
    n_timesteps: int,
    eval_interval: int,
    epsilon: float,
    num_repetitions: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    results = []
    for n_planning_updates in parameters:
        result = run_multiple_repetitions(
            spawn_env=spawn_env,
            spawn_agent=spawn_agent,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            epsilon=epsilon,
            n_planning_updates=n_planning_updates,
            num_repetitions=num_repetitions,
        )
        results.append(result)

    return results

def plot_deviation_learning_curves(inputs, eval_interval, title=None):
    plot = LearningCurvePlot(title=title or 'Mean evaluation return per timestep')
    for reward, label in inputs:
        mean, deviation = reward
        x = np.arange(len(mean)) * eval_interval
        plot.add_curve(mean, deviation=deviation if deviation is not None else None, label=label, x=x)

    return plot

def plot_learning_curves(inputs, eval_interval, title=None):
    plot = LearningCurvePlot(title=title or 'Mean evaluation return per timestep')
    for reward, label in inputs:
        mean = reward
        x = np.arange(len(mean)) * eval_interval
        plot.add_curve(mean, label=label, x=x)

    return plot

# Dyna
def run_dyna_planning_experiment(wind_proportion, n_timesteps, eval_interval, n_repetitions, gamma, learning_rate, epsilon, n_planning_updatess):
    print(f"Running Dyna planning sweep for wind_proportion = {wind_proportion}...")
    qlearning_results = run_multiple_repetitions(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=0,
        num_repetitions=n_repetitions,
    )

    results = run_experiment(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        parameters=n_planning_updatess,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        num_repetitions=n_repetitions,
    )

    curves = [(qlearning_results, "Q-learning baseline (n = 0)")]
    curves += [(mean, f"Dyna agent: n = {n}") for mean, n in zip(results, n_planning_updatess)]

    plot = plot_deviation_learning_curves(
        curves,
        eval_interval=eval_interval,
        title=f"Learning curve of the Dyna agent (wind = {wind_proportion})",
    )
    plot.save(f'dyna_learning_curve_wind_{wind_proportion}.png')

# Prioritized Sweeping
def run_ps_planning_experiment(wind_proportion, n_timesteps, eval_interval, n_repetitions, gamma, learning_rate, epsilon, n_planning_updatess):
    print(f"Running Prioritized Sweeping planning sweep for wind_proportion = {wind_proportion}...")
    qlearning_mean = run_multiple_repetitions(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=0,
        num_repetitions=n_repetitions,
    )

    means = run_experiment(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        parameters=n_planning_updatess,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        num_repetitions=n_repetitions,
    )

    curves = [(qlearning_mean, "Q-learning baseline (n = 0)")]
    curves += [(mean, f"Prioritized Sweeping agent: n = {n}") for mean, n in zip(means, n_planning_updatess)]

    plot = plot_deviation_learning_curves(
        curves,
        eval_interval=eval_interval,
        title=f"Learning curve of the Prioritized Sweeping agent (wind = {wind_proportion})",
    )
    plot.save(f'ps_learning_curve_wind_{wind_proportion}.png')

    return means

def best_index(means: List[np.ndarray]) -> int:
    # Select the curve with the highest mean over the second half of training
    scores = [np.mean(m[len(m) // 2:]) for m in means]
    return int(np.argmax(scores))

def experiment_comparison(wind_proportion, n_timesteps, eval_interval, n_repetitions, gamma, learning_rate, epsilon, n_planning_updatess):
    print(f"Running comparison experiment for wind_proportion = {wind_proportion}...")

    qlearning_mean = run_multiple_repetitions(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=0,
        num_repetitions=n_repetitions,
    )

    dyna_means = run_experiment(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        parameters=n_planning_updatess,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        num_repetitions=n_repetitions,
    )
    dyna_best_idx = best_index([mean for mean, _ in dyna_means])
    dyna_best_n = n_planning_updatess[dyna_best_idx]

    ps_means = run_experiment(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        parameters=n_planning_updatess,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        num_repetitions=n_repetitions,
    )
    ps_best_idx = best_index([mean for mean, _ in ps_means])
    ps_best_n = n_planning_updatess[ps_best_idx]

    plot = plot_deviation_learning_curves(
        [
            (qlearning_mean, "Q-learning baseline (n = 0)"),
            (dyna_means[dyna_best_idx], f"Dyna agent (n = {dyna_best_n})"),
            (ps_means[ps_best_idx], f"Prioritized Sweeping agent (n = {ps_best_n})"),
        ],
        eval_interval=eval_interval,
        title=f"Learning curve comparison (wind = {wind_proportion})",
    )
    plot.save(f'comparison_wind_{wind_proportion}.png')

def runtime_comparison(wind_proportion, n_timesteps, eval_interval, gamma, learning_rate, epsilon, n_planning_updates):
    print(f"Running runtime comparison for wind_proportion = {wind_proportion}...")

    configs = [
        ("Q-learning", lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma), 0),
        ("Dyna", lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma), n_planning_updates),
        ("Prioritized Sweeping", lambda env: PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma), n_planning_updates),
    ]

    print(f"\nRuntime comparison (wind_proportion = {wind_proportion}, n_planning_updates = {n_planning_updates}):")
    print(f"{'Algorithm':<25} {'Runtime (s)':>12}")
    print("-" * 39)
    for name, spawn_agent, n_planning in configs:
        start = time.time()
        run_single_repetition(
            spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
            spawn_agent=spawn_agent,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            epsilon=epsilon,
            n_planning_updates=n_planning,
        )
        elapsed = time.time() - start
        print(f"{name:<25} {elapsed:>12.2f}")
    print("-" * 39)

def single_run_comparison(wind_proportion, n_timesteps, eval_interval, gamma, learning_rate, epsilon, n_planning_updates):
    print(f"Running single run comparison for wind_proportion = {wind_proportion}...")

    qlearning_result = run_single_repetition(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=0,
    )

    dyna_result = run_single_repetition(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=n_planning_updates,
    )

    ps_result = run_single_repetition(
        spawn_env=lambda: WindyGridworld(wind_proportion=wind_proportion),
        spawn_agent=lambda env: PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma),
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        epsilon=epsilon,
        n_planning_updates=n_planning_updates,
    )

    plot = plot_learning_curves(
        [
            (qlearning_result, "Q-learning baseline (n = 0)"),
            (dyna_result, f"Dyna agent (n = {n_planning_updates})"),
            (ps_result, f"Prioritized Sweeping agent (n = {n_planning_updates})"),
        ],
        eval_interval=eval_interval,
        title=f"Single run comparison (wind = {wind_proportion})",
    )
    plot.save(f'single_run_comparison_wind_{wind_proportion}.png')

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 50
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]

    for wind_proportion in wind_proportions:
        single_run_comparison(
            wind_proportion=wind_proportion,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=epsilon,
            n_planning_updates=n_planning_updatess[-1],
        )

        run_dyna_planning_experiment(
            wind_proportion=wind_proportion,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            n_repetitions=n_repetitions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=epsilon,
            n_planning_updatess=n_planning_updatess,
        )
        run_ps_planning_experiment(
            wind_proportion=wind_proportion,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            n_repetitions=n_repetitions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=epsilon,
            n_planning_updatess=n_planning_updatess,
        )
        experiment_comparison(
            wind_proportion=wind_proportion,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            n_repetitions=n_repetitions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=epsilon,
            n_planning_updatess=n_planning_updatess,
        )
        runtime_comparison(
            wind_proportion=wind_proportion,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=epsilon,
            n_planning_updates=n_planning_updatess[-1],
        )

if __name__ == '__main__':
    experiment()
