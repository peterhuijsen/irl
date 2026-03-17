from typing import Any, Callable, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 

from scipy.signal import savgol_filter
from joblib import Parallel, delayed

from ShortCutAgents import Agent, QLearningAgent, SARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment

def smooth(y, window, poly=1):
    """
    Smooth a vector using a Savitzky-Golay filter.
    y: vector to be smoothed
    window: size of the smoothing window 
    """
    return savgol_filter(y,window,poly)

class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumulative reward')
        self.ax.set_ylim(bottom=-500, top=0)
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
            self.ax.legend()
        else:
            self.ax.plot(y)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def plot_cumulative_rewards(inputs):
    plot = LearningCurvePlot(title='Cumulative reward per episode')
    for reward, label in inputs:
        plot.add_curve(reward, label=label)
    return plot

def run_single_repetition(
    spawn_env: Callable[[], Environment], 
    spawn_agent: Callable[[Environment], Agent], 
    num_episodes=1000
):
    env = spawn_env()
    agent = spawn_agent(env)
    return agent.train(num_episodes)

def run_multiple_repetitions(
    spawn_env: Callable[[], Environment], 
    spawn_agent: Callable[[Environment], Agent], 
    num_repetitions: int, 
    num_episodes: int = 1000
):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_repetition)(spawn_env, spawn_agent, num_episodes)
        for _ in range(num_repetitions)
    )
    rewards = np.array(results)

    average_rewards = np.mean(rewards, axis=0) # average over the repetitions
    return smooth(average_rewards, window=31)

def run_experiment(
    spawn_env: Callable[[], Environment], 
    spawn_agent: Callable[[Environment, Any], Agent],
    parameters: List
):
    results = []
    for parameter in parameters:
        rewards = run_multiple_repetitions(
            spawn_env=spawn_env,
            spawn_agent=lambda env, p=parameter: spawn_agent(env, p),
            num_repetitions=100,
        )

        results.append(rewards)

    return results

def plot_q_values(Q, env):
    grid_shape = (env.r, env.c)
    actions = ['↑', '↓', '←', '→']

    color_grid = np.zeros(grid_shape)
    for y in range(env.r):
        for x in range(env.c):
            if env.s[y, x] == 'C': color_grid[y, x] = 1 # If its wall
            elif env.s[y, x] == 'G': color_grid[y, x] = 2 # If its goal
            else: color_grid[y, x] = 0 # If its empty cell

    start_coords = [(2,2), (9,2)]
    for (y, x) in start_coords:
        color_grid[y, x] = 3 #mark start positions
    
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = mcolors.ListedColormap(['white', 'lightcoral', 'lightgreen', 'lightblue'])
    ax.matshow(color_grid, cmap=cmap)

    ax.set_xticks(np.arange(-0.5, env.c, 1), minor =True)
    ax.set_yticks(np.arange(-0.5, env.r, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])


    path_colors = ['blue', 'orange']
    cells_in_path = set()
    paths_to_draw = []

    for idx, (start_y, start_x) in enumerate(start_coords):
        curr_y, curr_x = start_y, start_x
        path_x, path_y = [curr_x], [curr_y]
        visited = set()
        while env.s[curr_y, curr_x] not in ['G', 'C']:
            if (curr_y, curr_x) in visited:
                break

            visited.add((curr_y, curr_x))
            cells_in_path.add((curr_y, curr_x))

            state = curr_y * env.c + curr_x
            best_action = np.argmax(Q[state])

            if best_action == 0 and curr_y > 0: curr_y -= 1 # Up
            elif best_action == 1 and curr_y < env.r - 1: curr_y += 1 # Down
            elif best_action == 2 and curr_x > 0: curr_x -= 1 # Left
            elif best_action == 3 and curr_x < env.c - 1: curr_x += 1 # Right
            path_x.append(curr_x)
            path_y.append(curr_y)
        
        cells_in_path.add((curr_y, curr_x))
        paths_to_draw.append((path_x, path_y, path_colors[idx], f'Path from Start {idx+1}'))
    
    for y in range(env.r):
        for x in range(env.c):
            if env.s[y, x] in ['C', 'G'] or (y, x) in cells_in_path: continue # Skip walls and goal
            q_values = Q[y * env.c + x]
            if np.max(q_values) != 0 or np.sum(q_values) != 0: 
                best_action = np.argmax(q_values)
                action_symbol = actions[best_action] 
                ax.text(x, y, action_symbol, ha='center', va='center', fontsize=16)


    for path_x, path_y, color, label in paths_to_draw:
        ax.plot(path_x, path_y, color=color, linewidth=3, alpha=0.7, marker='o', label=label)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Optimal Actions and Paths from Start to Goal')
    plt.tight_layout()  
    plt.show()

def run_qlearning_experiment_with_different_alphas(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env, alpha: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
    )

    plot = plot_cumulative_rewards([(reward, f"QLearningAgent: α = {alpha}") for reward, alpha in zip(rewards, alphas)])
    plot.save('Figures/qlearning_alpha_comparison.png')
    
def run_sarsa_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    plot = plot_cumulative_rewards([(rewards, "SARSAAgent")])
    plot.save('Figures/sarsa.png')

def run_sarsa_experiment_with_different_alphas(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env, alpha: SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
    )

    plot = plot_cumulative_rewards([(reward, f"SARSAAgent: α = {alpha}") for reward, alpha in zip(rewards, alphas)])
    plot.save('Figures/sarsa_alpha_comparison.png')

def run_qlearning_vs_sarsa_experiment(environment_class):
    qlearning_rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    plot = plot_cumulative_rewards([
        (qlearning_rewards, "QLearningAgent"),
        (sarsa_rewards, "SARSAAgent")
    ])
    plot.save('Figures/qlearning_vs_sarsa_comparison.png')
    
def experiments(environment_class):
    print(f"Running QLearning experiment for {environment_class.__name__}...")
    run_qlearning_experiment(environment_class=environment_class)

    print(f"Running QLearning experiment with different alphas for {environment_class.__name__}...")
    run_qlearning_experiment_with_different_alphas(environment_class=environment_class)

    print(f"Running SARSAAgent experiment for {environment_class.__name__}...")
    run_sarsa_experiment(environment_class=environment_class)

    print(f"Running SARSAAgent experiment with different alphas for {environment_class.__name__}...")
    run_sarsa_experiment_with_different_alphas(environment_class=environment_class)

    print(f"Running QLearning vs SARSAAgent experiment for {environment_class.__name__}...")
    run_qlearning_vs_sarsa_experiment(environment_class=environment_class)

if __name__ == "__main__":
    experiments(environment_class=ShortcutEnvironment)
