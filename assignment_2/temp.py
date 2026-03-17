from typing import Any, Callable, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 

from scipy.signal import savgol_filter
from joblib import Parallel, delayed

from ShortCutAgents import Agent, QLearningAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment

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
    plot.save()

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
        delayed(lambda: run_single_repetition(spawn_env, spawn_agent, num_episodes))()
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
            spawn_agent=lambda env: spawn_agent(env, parameter),
            num_repetitions=100,
        )

        results.append(rewards)

    return results

def run_qlearning_experiment():
    rewards = run_multiple_repetitions(
        spawn_env=lambda: ShortcutEnvironment(),
        spawn_agent=lambda env: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    plot_cumulative_rewards([rewards, "QLearningAgent"])

def run_qlearning_experiment_with_different_alphas():
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda: ShortcutEnvironment(),
        spawn_agent=lambda env, alpha: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.05, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
    )

    plot_cumulative_rewards([(reward, f"QLearningAgent: α = {alpha}") for reward, alpha in zip(rewards, alphas)])
    
run_qlearning_experiment_with_different_alphas()

# Q = np.zeros([env.state_size(),env.action_size()])    
def plot_q_values(Q, env):
    grid_shape = (env.r, env.c)
    actions = ['↑', '↓', '←', '→']

    greedy_action = np.argmax(Q, axis=1).reshape(grid_shape) # Reshape to match the grid
    max_q_values = np.max(Q, axis=1).reshape(grid_shape) # Reshape to match the grid
    color_grid = np.zeros(grid_shape)
    for y in range(env.r):
        for x in range(env.c):
            if env.s[y, x] == 'C': color_grid[y, x] = 1 # If it's a wall
            elif env.s[y, x] == 'G': color_grid[y, x] = 2 # If it's the goal
            else: color_grid[y, x] = 0 # If it's an empty cell
    start_coords = [(2,2), (9,2)]
    for (y, x) in start_coords:
        color_grid[y, x] = 3 # Mark the start positions
    
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = mcolors.ListedColormap(['white', 'lightcoral', 'lightgreen', 'lightblue'])
    ax.matshow(color_grid, cmap=cmap)

    ax.set_xticks(np.arange(-0.5, env.c, 1), minor =True)
    ax.set_yticks(np.arange(-0.5, env.r, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    path_colors = ['blue', 'orange']
    optimal_paths = set()
    paths_to_draw = []

    for idx, (start_y, start_x) in enumerate(start_coords):
        curr_y, curr_x = start_y, start_x
        path_x, path_y = [curr_x], [curr_y]
        visited = set()
        while env.s[curr_y, curr_x] != 'G' and env.s[curr_y, curr_x] != 'C':
            if (curr_y, curr_x) in visited:
                break
            visited.add((curr_y, curr_x))
            optimal_paths.add((curr_y, curr_x))
            action = greedy_action[curr_y, curr_x]
            if action == 0 and curr_y > 0: curr_y -= 1 # Up
            elif action == 1 and curr_y < env.r - 1: curr_y += 1 # Down
            elif action == 2 and curr_x > 0: curr_x -= 1 # Left
            elif action == 3 and curr_x < env.c - 1: curr_x += 1 # Right
            path_x.append(curr_x)
            path_y.append(curr_y)
        paths_to_draw.append((path_x, path_y))
        path_y, path_colors[idx], label = path_y, path_colors[idx], f'Path from start {idx+1}'


    for y in range(env.r):
        for x in range(env.c):
            if env.s[y, x] in ['C', 'G']: continue # Skip walls and goal
            q_values = Q[y * env.c + x]
            best_action = np.argmax(q_values)
            action_symbol = actions[best_action][0] # First letter of the action
            ax.text(x, y, action_symbol, ha='center', va='center', fontsize=12)
    path_colors = ['blue', 'orange']
    
        path.append((current_state // env.c, current_state % env.c)) # Add the goal state
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], color=path_colors[idx], linewidth=2, label=f'Path from start {idx+1}')
        
    ax.legend(loc='upper right')
    plt.title('Optimal Actions and Paths from Start to Goal')
    plt.show()

if __name__ == "__main__":
    env = ShortcutEnvironment()
    agent = QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env)
    agent.train(n_episodes=1000)
    plot_q_values(agent.Q, env)