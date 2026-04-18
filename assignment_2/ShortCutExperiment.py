from enum import Enum
from typing import Any, Callable, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 

from scipy.signal import savgol_filter
from joblib import Parallel, delayed

from ShortCutAgents import Agent, DecayingExpectedSARSAAgent, ExpectedSARSAAgent, QLearningAgent, SARSAAgent, nStepSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment

global_seed = 42

class AgentType(Enum):
    QLearning = "QLearning"
    SARSA = "SARSA"
    ExpectedSARSA = "Expected SARSA"
    NStepSARSA = "n-Step SARSA"

class ExecutionType(Enum):
    Single = "Single"
    Multiple = "Multiple"
    Path = "Path"

class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumulative reward')
        self.ax.set_ylim(bottom=-500, top=0)
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, mean, deviation=None, label=None):
        if deviation is not None:
            upper_bound = mean + deviation
            lower_bound = mean - deviation
            self.ax.fill_between(range(len(mean)), lower_bound, upper_bound, alpha=0.1, label=f"_hidden_{label}_deviation")
        
        if label is not None:
            self.ax.plot(mean, label=label)
            self.ax.legend()
        else:
            self.ax.plot(mean)


    def save(self,name='test.png'):
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

class GreedyPathPlot:
    def __init__(self, title=None):
        self.figure, self.ax = plt.subplots(figsize=(12, 12))
        self.starts = [(2,2), (9,2)]
        if title is not None:
            self.ax.set_title(title, fontsize=18)

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
        
        self.ax.matshow(
            color_grid, 
            cmap=mcolors.ListedColormap(['white', 'lightcoral', 'lightgreen', 'lightblue'])
        )

        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, environment.c, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, environment.r, 1), minor=True)

    def add_agent(self, agent: Agent):
        if not self.environment:
            raise Exception("Need to initialize environment before adding an agent")
        
        actions = ['↑', '↓', '←', '→']
        path_colors = ['blue', 'orange']
        cells_in_path = set()

        # Draw the paths
        for i, (y, x) in enumerate(self.starts):
            path = [(x, y)]
            step_count = 0
            while self.environment.s[y, x] not in ['G', 'C'] and step_count < 100:
                cells_in_path.add((y, x))

                state = y * self.environment.c + x

                # Since we want to visualize the greedy path we do
                # not use the action selection from the agent itself
                # since that would be an epsilon-greedy action selection
                # where then agent also explores.
                best_action = np.argmax(agent.Q[state])

                self.environment.x, self.environment.y = x, y

                self.environment.isdone = False
                self.environment.step(best_action)

                x, y = self.environment.x, self.environment.y

                path.append((x, y))
                step_count += 1
            
            cells_in_path.add((y, x))
            self.ax.plot(
                [point[0] for point in path], 
                [point[1] for point in path], 
                color=path_colors[i],
                linewidth=3, 
                alpha=0.7, 
                marker='o', 
                label=f'Path from Start {i+1}'
            )

        # Draw the arrows
        for y in range(self.environment.r):
            for x in range(self.environment.c):
                # Skip walls, goals, and path cells
                if self.environment.s[y, x] in ['C', 'G'] or (y, x) in cells_in_path: 
                    continue 
                
                greedy_path = agent.Q[y * self.environment.c + x]
                if np.max(greedy_path) != 0 or np.sum(greedy_path) != 0: 
                    best_action = np.argmax(greedy_path)
                    action_symbol = actions[best_action] 
                    self.ax.text(x, y, action_symbol, ha='center', va='center', fontsize=24)

        self.ax.legend(loc='upper right')

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.figure.savefig(name, dpi=300, bbox_inches='tight')

def plot_cumulative_rewards(inputs, title=None, variance=True):
    plot = LearningCurvePlot(title=title or 'Cumulative reward per episode')
    for reward, label in inputs:
        mean, deviation = reward
        plot.add_curve(mean, deviation=deviation if variance else None, label=label)

    return plot

def smooth(y, window, poly=1) -> np.ndarray:
    """
    Smooth a vector using a Savitzky-Golay filter.
    y: vector to be smoothed
    window: size of the smoothing window 
    """
    return np.array(savgol_filter(y,window,poly))

def run_single_repetition(
    spawn_env: Callable[[int], Environment], 
    spawn_agent: Callable[[Environment], Agent], 
    num_episodes=1000,
    seed=global_seed
) -> np.ndarray:
    env = spawn_env(seed)
    agent = spawn_agent(env)
    return agent.train(num_episodes)

def run_multiple_repetitions(
    spawn_env: Callable[[int], Environment], 
    spawn_agent: Callable[[Environment], Agent], 
    num_repetitions: int = 100, 
    num_episodes: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    results = Parallel(n_jobs=-1)(
        delayed(run_single_repetition)(spawn_env, spawn_agent, num_episodes, seed=global_seed + i)
        for i in range(num_repetitions)
    )
    rewards = np.array(results)

    average_rewards = np.mean(rewards, axis=0)
    sd_rewards = np.std(rewards, axis=0)
    return smooth(average_rewards, window=31), smooth(sd_rewards, window=31)

def run_experiment(
    spawn_env: Callable[[int], Environment], 
    spawn_agent: Callable[[Environment, Any], Agent],
    parameters: List,
    num_repetitions=100,
    num_episodes=1000
) -> List[Tuple[np.ndarray, np.ndarray]]:
    results = []
    for parameter in parameters:
        mean, deviation = run_multiple_repetitions(
            spawn_env=spawn_env,
            spawn_agent=lambda env, p=parameter: spawn_agent(env, p),
            num_repetitions=num_repetitions,
            num_episodes=num_episodes
        )

        results.append((mean, deviation))

    return results

def plot_greedy_path(agent: Agent, environment: Environment, title="Greedy paths"):
    plot = GreedyPathPlot(title=title)
    plot.add_environment(environment=environment)
    plot.add_agent(agent=agent)
    return plot

# QLearning
def run_qlearning_greedy_path_experiment(environment_class):
    env = environment_class(seed=global_seed)
    agent = QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_greedy_path(agent=agent, environment=env, title="Greedy path of the Q-learning agent")
    plot.save("qlearning_greedy_path.png")

def run_qlearning_lc_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(rewards, "Q-learning agent")], title="Learning curve of Q-learning agent")
    plot.save('qlearning_learning_curve.png')

def run_qlearning_lc_alphas_experiment(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env, alpha: QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards(
        [(reward, f"Q-learning agent: α = {alpha}") for reward, alpha in zip(rewards, alphas)], 
        title="Learning curve comparison of the Q-learning agent"
    )
    plot.save('qlearning_learning_curve_alphas.png')
    
# SARSA
def run_sarsa_greedy_path_experiment(environment_class):
    env = environment_class(seed=global_seed)
    agent = SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_greedy_path(agent=agent, environment=env, title="Greedy path of the SARSA agent")
    plot.save("sarsa_greedy_path.png")

def run_sarsa_lc_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(rewards, "SARSA agent")], title="Learning curve of the SARSA agent")
    plot.save('sarsa_learning_curve.png')

def run_sarsa_lc_alphas_experiment(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env, alpha: SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards(
        [(reward, f"SARSA agent: α = {alpha}") for reward, alpha in zip(rewards, alphas)], 
        title="Learning curve comparison of the SARSA agent"
    )
    plot.save('sarsa_learning_curve_alphas.png')

# Expected SARSA
def run_expected_sarsa_greedy_path_experiment(environment_class):
    env = environment_class(seed=global_seed)
    agent = ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env)
    agent.train(n_episodes=10000)
    plot = plot_greedy_path(agent=agent, environment=env, title="Greedy path of the expected SARSA agent")
    plot.save("expected_sarsa_greedy_path.png")

def run_expected_sarsa_lc_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(rewards, "Expected SARSA agent")], title="Learning curve of the expected SARSA agent")
    plot.save('expected_sarsa_learning_curve.png')

def run_expected_sarsa_lc_alphas_experiment(environment_class):
    alphas = [0.01, 0.1, 0.5, 0.9]
    rewards = run_experiment(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env, alpha: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env),
        parameters=alphas,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards(
        [(reward, f"Expected SARSA agent: α = {alpha}") for reward, alpha in zip(rewards, alphas)], 
        title="Learning curve comparison of the expected SARSA agent"
    )
    plot.save('expected_sarsa_learning_curve_alphas.png')

# n-Step SARSA
def run_n_step_sarsa_greedy_path_experiment(environment_class):
    env = environment_class(seed=global_seed)
    agent = nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=env, n=5)
    agent.train(n_episodes=10000)
    plot = plot_greedy_path(agent=agent, environment=env, title="Greedy path of the n-step SARSA agent")
    plot.save("n_step_sarsa_greedy_path.png")

def run_n_step_sarsa_experiment(environment_class):
    rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env, n=5),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(rewards, "n-step SARSA agent")], title="Learning curve of the n-step SARSA agent")
    plot.save('n_step_sarsa_learning_curve.png')

def run_n_step_sarsa_experiment_with_different_ns(environment_class):
    ns = [1, 2, 5, 10, 25]
    rewards = run_experiment(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env, n: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env, n=n),
        parameters=ns,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards(
        [(reward, f"n-step SARSA agent: n = {n}") for reward, n in zip(rewards, ns)],
        title="Learning curve comparison of the n-step SARSA agent"
    )
    plot.save('n_step_sarsa_learning_curve_ns.png')

def run_n_step_sarsa_experiment_with_different_ns_high_a(environment_class):
    alpha = 0.5
    ns = [1, 2, 5, 10, 25]
    rewards = run_experiment(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env, n: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=alpha, gamma=1.0, env=env, n=n),
        parameters=ns,
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards(
        [(reward, f"n-step SARSA agent: n = {n}") for reward, n in zip(rewards, ns)],
        title=fr"Learning curve comparison of the n-step SARSA agent ($\alpha$ = {alpha})"
    )
    plot.save('n_step_sarsa_learning_curve_ns_high_a.png')

# Windy environment
def experiment_windy():
    q_env, s_env = WindyShortcutEnvironment(seed=41), WindyShortcutEnvironment(seed=41)
    print(f"Running greedy path visualization for Q-learning and SARSA agents in {q_env.__class__.__name__}...")
    q_learning = QLearningAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=q_env)
    sarsa = SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1, env=s_env)

    q_learning.train(n_episodes=10000)
    sarsa.train(n_episodes=10000)
    
    q_learning_plot, sarsa_plot = plot_greedy_path(agent=q_learning, environment=q_env, title="Greedy path of the Q-learning agent (windy)"), plot_greedy_path(agent=sarsa, environment=s_env, title="Greedy path of the SARSA agent (windy)")
    q_learning_plot.save("windy_shortcut_qlearning_agent.png")
    sarsa_plot.save("windy_shortcut_sarsa_agent.png")

def experiment(agents: Dict[AgentType, Set[ExecutionType]], environment_class):
    def experiment_qlearning():
        if AgentType.QLearning not in agents:
            return
                
        execution_type = agents[AgentType.QLearning]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of QLearning experiment for {environment_class.__name__}...")
            run_qlearning_lc_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of QLearning experiment for {environment_class.__name__}...")
            run_qlearning_lc_alphas_experiment(environment_class=environment_class)
        if ExecutionType.Path in execution_type:
            print(f"Running greedy path visualization for QLearning agent in {environment_class.__name__}...")
            run_qlearning_greedy_path_experiment(environment_class=environment_class)

    def experiment_sarsa():
        if AgentType.SARSA not in agents:
            return
        
        execution_type = agents[AgentType.SARSA]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of SARSAAgent experiment for {environment_class.__name__}...")
            run_sarsa_lc_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of SARSAAgent experiment for {environment_class.__name__}...")
            run_sarsa_lc_alphas_experiment(environment_class=environment_class)
        if ExecutionType.Path in execution_type:
            print(f"Running greedy path visualization for SARSAAgent in {environment_class.__name__}...")
            run_sarsa_greedy_path_experiment(environment_class=environment_class)

    def experiment_expected_sarsa():
        if AgentType.ExpectedSARSA not in agents:
            return
        
        execution_type = agents[AgentType.ExpectedSARSA]
        if ExecutionType.Single in execution_type:
            print(f"Running single repetition of ExpectedSARSAAgent experiment for {environment_class.__name__}...")
            run_expected_sarsa_lc_experiment(environment_class=environment_class)
        if ExecutionType.Multiple in execution_type:
            print(f"Running multiple repetitions of ExpectedSARSAAgent experiment for {environment_class.__name__}...")
            run_expected_sarsa_lc_alphas_experiment(environment_class=environment_class)
        if ExecutionType.Path in execution_type:
            print(f"Running greedy path visualization for ExpectedSARSAAgent in {environment_class.__name__}...")
            run_expected_sarsa_greedy_path_experiment(environment_class=environment_class)

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
        if ExecutionType.Path in execution_type:
            print(f"Running greedy path visualization for n-Step SARSAAgent in {environment_class.__name__}...")
            run_n_step_sarsa_greedy_path_experiment(environment_class=environment_class)

    experiment_qlearning()
    experiment_sarsa()
    experiment_expected_sarsa()
    experiment_n_step_sarsa()

def experiment_comparison(environment_class):
    print(f"Running comparison experiment for {environment_class.__name__}...")
    n, alpha, epsilon, gamma = 5, 0.1, 0.1, 1.0
    repetitions, episodes = 100, 1000

    qlearning_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: QLearningAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env),
        num_repetitions=repetitions,
        num_episodes=episodes
    )

    sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: SARSAAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env),
        num_repetitions=repetitions,
        num_episodes=episodes
    )
    
    expected_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env),
        num_repetitions=repetitions,
        num_episodes=episodes
    )

    nstep_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env, n=n),
        num_repetitions=repetitions,
        num_episodes=episodes
    )

    plot = plot_cumulative_rewards([
        (qlearning_rewards, "Q-learning agent"),
        (sarsa_rewards, "SARSA agent"),
        (expected_sarsa_rewards, "Expected SARSA agent"),
        (nstep_sarsa_rewards, "n-Step SARSA agent"),
    ], title="Learning curve comparison")
    plot.save('qlearning_vs_sarsa_vs_expected_sarsa_comparison.png')

def experiment_extension_greedy_path(environment_class):
    print(f"Running extension greedy path experiment for {environment_class.__name__}...")
    env = environment_class(seed=global_seed)
    agent = DecayingExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=1.0, alpha=0.1, gamma=1, env=env, decay_rate=0.8)
    agent.train(n_episodes=10000)
    plot = plot_greedy_path(agent=agent, environment=env, title=r"Greedy path of decaying-$\varepsilon$ expected SARSA agent")
    plot.save("extension_greedy_path.png")

def experiment_extension_learning_curve(environment_class):
    print(f"Running extension learning curve experiment for {environment_class.__name__}...")
    rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: DecayingExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=1.0, alpha=0.1, gamma=1.0, env=env, decay_rate=0.80),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([(rewards, r"Decaying-$\varepsilon$ expected SARSA agent")], title=r"Learning curve of decaying-$\varepsilon$ expected SARSA agent")
    plot.save('extension_learning_curve_decay_rates.png')

def experiment_extension_comparison(environment_class):
    print(f"Running extension comparison experiment for {environment_class.__name__}...")

    n, alpha, epsilon, gamma = 5, 0.1, 0.1, 1.0
    repetitions, episodes = 100, 1000
    
    expected_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: ExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env),
        num_repetitions=repetitions,
        num_episodes=episodes
    )

    nstep_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: nStepSARSAAgent(n_actions=4, n_states=12**2, epsilon=epsilon, alpha=alpha, gamma=gamma, env=env, n=n),
        num_repetitions=repetitions,
        num_episodes=episodes
    )

    decaying_sarsa_rewards = run_multiple_repetitions(
        spawn_env=lambda seed: environment_class(seed=seed),
        spawn_agent=lambda env: DecayingExpectedSARSAAgent(n_actions=4, n_states=12**2, epsilon=1.0, alpha=0.1, gamma=1.0, env=env, decay_rate=0.80),
        num_repetitions=100,
        num_episodes=1000
    )

    plot = plot_cumulative_rewards([
        (expected_sarsa_rewards, "Expected SARSA agent"),
        (nstep_sarsa_rewards, "n-step SARSA agent"),
        (decaying_sarsa_rewards, r"Decaying-$\varepsilon$ expected SARSA agent"),
    ], title="Learning curve extension comparison")
    plot.save('extension_comparison.png')

def experiment_extension(environment_class):
    experiment_extension_greedy_path(environment_class=environment_class)
    experiment_extension_learning_curve(environment_class=environment_class)
    experiment_extension_comparison(environment_class=environment_class)

if __name__ == "__main__":
    experiment_windy()
    experiment(
        agents={
            AgentType.QLearning: { 
                ExecutionType.Single, 
                ExecutionType.Multiple, 
                ExecutionType.Path
            },
            AgentType.SARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple, 
                ExecutionType.Path
            },
            AgentType.ExpectedSARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple,
                ExecutionType.Path
            },
            AgentType.NStepSARSA: { 
                ExecutionType.Single, 
                ExecutionType.Multiple,
                ExecutionType.Path
            },
        },
        environment_class=ShortcutEnvironment
    )
    experiment_comparison(environment_class=ShortcutEnvironment)

    # # Extra experiment for n-step SARSA
    run_n_step_sarsa_experiment_with_different_ns_high_a(environment_class=ShortcutEnvironment)
   
    experiment_extension(environment_class=ShortcutEnvironment)