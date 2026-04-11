from enum import Enum
from typing import Any, Callable, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 

from scipy.signal import savgol_filter
from joblib import Parallel, delayed

from ShortCutAgents import Agent, ExpectedSARSAAgent, QLearningAgent, SARSAAgent, nStepSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment

class AgentType(Enum):
    QLearning = "QLearning"
    SARSA = "SARSA"
    ExpectedSARSA = "Expected SARSA"
    NStepSARSA = "n-Step SARSA"

class ExecutionType(Enum):
    Single = "Single"
    Multiple = "Multiple"
    Q = "Q"

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

class QValuesPlot:
    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.starts = [(2,2), (9,2)]
        if title is not None:
            self.ax.set_title(title)

    def add_environment(self, environment):
        self.environment = environment
        grid_shape = (environment.r, environment.c)
        color_grid = np.zeros(grid_shape)
        for y in range(environment.r):
            for x in range(environment.c):
                if environment.s[y, x] == 'C': color_grid[y, x] = 1 # If its wall
                elif environment.s[y, x] == 'G': color_grid[y, x] = 2 # If its goal
                else: color_grid[y, x] = 0 # If its empty cell

        for (y, x) in self.starts:
            color_grid[y, x] = 3 #mark start positions
        
        cmap = mcolors.ListedColormap(['white', 'lightcoral', 'lightgreen', 'lightblue'])
        self.ax.matshow(color_grid, cmap=cmap)

        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, environment.c, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, environment.r, 1), minor=True)

    def add_agent(self, agent: Agent):
        if not self.environment:
            raise Exception("Need to initialize environment before adding an agent")
        
        actions = ['↑', '↓', '←', '→']
        path_colors = ['blue', 'orange']
        cells_in_path = set()
        paths_to_draw = []

        for idx, (start_y, start_x) in enumerate(self.starts):
            curr_y, curr_x = start_y, start_x
            path_x, path_y = [curr_x], [curr_y]
            visited = set()
            while self.environment.s[curr_y, curr_x] not in ['G', 'C']:
                if (curr_y, curr_x) in visited:
                    break

                visited.add((curr_y, curr_x))
                cells_in_path.add((curr_y, curr_x))

                state = curr_y * self.environment.c + curr_x
                best_action = np.argmax(agent.Q[state])

                if best_action == 0 and curr_y > 0: curr_y -= 1 # Up
                elif best_action == 1 and curr_y < self.environment.r - 1: curr_y += 1 # Down
                elif best_action == 2 and curr_x > 0: curr_x -= 1 # Left
                elif best_action == 3 and curr_x < self.environment.c - 1: curr_x += 1 # Right
                path_x.append(curr_x)
                path_y.append(curr_y)
            
            cells_in_path.add((curr_y, curr_x))
            paths_to_draw.append((path_x, path_y, path_colors[idx], f'Path from Start {idx+1}'))

        for y in range(self.environment.r):
            for x in range(self.environment.c):
                if self.environment.s[y, x] in ['C', 'G'] or (y, x) in cells_in_path: continue # Skip walls and goal
                q_values = agent.Q[y * self.environment.c + x]
                if np.max(q_values) != 0 or np.sum(q_values) != 0: 
                    best_action = np.argmax(q_values)
                    action_symbol = actions[best_action] 
                    self.ax.text(x, y, action_symbol, ha='center', va='center', fontsize=16)

        for path_x, path_y, color, label in paths_to_draw:
            self.ax.plot(path_x, path_y, color=color, linewidth=3, alpha=0.7, marker='o', label=label)
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

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
    num_repetitions: int=100, 
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
    parameters: List,
    num_repetitions=100,
    num_episodes=1000
):
    results = []
    for parameter in parameters:
        rewards = run_multiple_repetitions(
            spawn_env=spawn_env,
            spawn_agent=lambda env, p=parameter: spawn_agent(env, p),
            num_repetitions=num_repetitions,
            num_episodes=num_episodes
        )

        results.append(rewards)

    return results

def plot_q_values(agent: Agent, environment: Environment, title="Q-values"):
    plot = QValuesPlot(title=title)
    plot.add_environment(environment=environment)
    plot.add_agent(agent=agent)
    return plot

def run_qlearning_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    plot = plot_cumulative_rewards([(rewards, "QLearningAgent")])
    plot.save('Figures/qlearning.png')

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

def run_expected_sarsa_experiment(environment_class):
    rewards = run_single_repetition(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_episodes=10000
    )

    plot = plot_cumulative_rewards([(rewards, "ExpectedSARSAAgent")])
    plot.save('Figures/expected_sarsa.png')

def run_expected_sarsa_experiment_with_different_alphas(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env, alpha: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(reward, f"ExpectedSARSAAgent: α = {alpha}") for reward, alpha in zip(rewards, alphas)])
    plot.save('Figures/expected_sarsa_alpha_comparison.png')

def run_n_step_sarsa_experiment(environment_class):
    rewards = run_single_repetition(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env, n=5),
        num_episodes=10000
    )

    plot = plot_cumulative_rewards([(rewards, "n-Step SARSAAgent")])
    plot.save('Figures/n_step_sarsa.png')

def run_n_step_sarsa_experiment_with_different_ns(environment_class):
    ns = [1, 2, 5, 10, 25]
    rewards = run_experiment(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env, n: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env, n=n),
        parameters=ns,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(reward, f"n-Step SARSAAgent: n = {n}") for reward, n in zip(rewards, ns)])
    plot.save('Figures/n_step_sarsa_n_comparison.png')
   
def run_qlearning_plot_q_values(environment_class):
    env = environment_class()
    agent = QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_q_values(agent=agent, environment=env, title="QLearning agent")
    plot.save("Figures/qlearning_agent.png")

def run_sarsa_plot_q_values(environment_class):
    env = environment_class()
    agent = SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_q_values(agent=agent, environment=env, title="SARSA agent")
    plot.save("Figures/sarsa_agent.png")

def run_expected_sarsa_plot_q_values(environment_class):
    env = environment_class()
    agent = ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_q_values(agent=agent, environment=env, title="Expected SARSA agent")
    plot.save("Figures/expected_sarsa_agent.png")

def run_n_step_sarsa_plot_q_values(environment_class):
    env = environment_class()
    agent = nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env, n=5)
    agent.train(n_episodes=10000)
    plot = plot_q_values(agent=agent, environment=env, title="n-Step SARSA agent")
    plot.save("Figures/n_step_sarsa_agent.png")

def experiment(agents: Dict[AgentType, Set[ExecutionType]], environment_class):
    def experiment_qlearning():
        if AgentType.QLearning not in agents:
            return
                
        execution_type = agents[AgentType.QLearning]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of QLearning experiment for {environment_class.__name__}...")
            run_qlearning_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of QLearning experiment for {environment_class.__name__}...")
            run_qlearning_experiment_with_different_alphas(environment_class=environment_class)
        if ExecutionType.Q in execution_type:
            print(f"Running Q-value visualization for QLearning agent in {environment_class.__name__}...")
            run_qlearning_plot_q_values(environment_class=environment_class)

    def experiment_sarsa():
        if AgentType.SARSA not in agents:
            return
        
        execution_type = agents[AgentType.SARSA]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of SARSAAgent experiment for {environment_class.__name__}...")
            run_sarsa_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of SARSAAgent experiment for {environment_class.__name__}...")
            run_sarsa_experiment_with_different_alphas(environment_class=environment_class)
        if ExecutionType.Q in execution_type:
            print(f"Running Q-value visualization for SARSAAgent in {environment_class.__name__}...")
            run_sarsa_plot_q_values(environment_class=environment_class)

    def experiment_expected_sarsa():
        if AgentType.ExpectedSARSA not in agents:
            return
        
        execution_type = agents[AgentType.ExpectedSARSA]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of ExpectedSARSAAgent experiment for {environment_class.__name__}...")
            run_expected_sarsa_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of ExpectedSARSAAgent experiment for {environment_class.__name__}...")
            run_expected_sarsa_experiment_with_different_alphas(environment_class=environment_class)
        if ExecutionType.Q in execution_type:
            print(f"Running Q-value visualization for ExpectedSARSAAgent in {environment_class.__name__}...")
            run_expected_sarsa_plot_q_values(environment_class=environment_class)

    def experiment_n_step_sarsa():
        if AgentType.NStepSARSA not in agents:
            return
        
        execution_type = agents[AgentType.NStepSARSA]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of n-Step SARSAAgent experiment for {environment_class.__name__}...")
            run_n_step_sarsa_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of n-Step SARSAAgent experiment for {environment_class.__name__}...")
            run_n_step_sarsa_experiment_with_different_ns(environment_class=environment_class)
        if ExecutionType.Q in execution_type:
            print(f"Running Q-value visualization for n-Step SARSAAgent in {environment_class.__name__}...")
            run_n_step_sarsa_plot_q_values(environment_class=environment_class)

    experiment_qlearning()
    experiment_sarsa()
    experiment_expected_sarsa()
    experiment_n_step_sarsa()

def experiment_comparison(environment_class):
    print(f"Running comparison experiment for {environment_class.__name__}...")
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
    
    expected_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
    )

    nstep_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda: environment_class(),
        spawn_agent=lambda env: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env, n=2),
        num_repetitions=100,
    )

    plot = plot_cumulative_rewards([
        (qlearning_rewards, "QLearningAgent"),
        (sarsa_rewards, "SARSAAgent"),
        (expected_sarsa_rewards, "ExpectedSARSAAgent"),
        (nstep_sarsa_rewards, "n-Step SARSAAgent"),
    ])
    plot.save('Figures/qlearning_vs_sarsa_vs_expected_sarsa_comparison.png')

if __name__ == "__main__":
    experiment_comparison(environment_class=ShortcutEnvironment)
    experiment(
        agents={
            AgentType.QLearning: { 
                ExecutionType.Single, 
                ExecutionType.Multiple, 
                ExecutionType.Q
            },
            AgentType.SARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple, 
                ExecutionType.Q
            },
            AgentType.ExpectedSARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple,
                ExecutionType.Q
            },
            AgentType.NStepSARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple,
                ExecutionType.Q
            },
        },
        environment_class=ShortcutEnvironment
    )
