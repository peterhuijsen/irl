import numpy as np

from ShortCutEnvironment import ShortcutEnvironment

class Agent(object):
    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # Get the best current action.
        best_action = np.argmax(self.Q[state])
        policy = []

        # For every action, we determine the probability of choosing it and add it 
        # to the current policy.
        for action in range(self.n_actions):
            action_probability = 1 - self.epsilon if action == best_action else \
                self.epsilon / (self.n_actions - 1)
            policy.append(action_probability)

        # Then we randomly choose one of the actions based on the pdf.
        action = np.random.choice(self.n_actions, p=policy)
        return action
     
    def train(self, n_episodes) -> list:
        raise NotImplementedError("This method should be implemented by subclasses.")
        

class QLearningAgent(Agent):
    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0, env=ShortcutEnvironment()):
        super().__init__(n_actions, n_states, epsilon, alpha, gamma)
        self.env = env
        
    def update(self, state, action, reward, next_state): # Augment arguments if necessary
        # Q-learning update rule
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            self.env.reset()
            state = self.env.state()
            cumulative_reward = 0

            while not self.env.done():
                # Select the action based on the policy
                action = self.select_action(state)

                # Update the environment and get the reward
                reward = self.env.step(action)

                # Update the Q-table based on the transition
                s_next = self.env.state()
                self.update(state, action, reward, s_next)

                state = s_next
                cumulative_reward += reward

            episode_returns.append(cumulative_reward)

        return episode_returns


class SARSAAgent(Agent):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0, env=ShortcutEnvironment()):
        super().__init__(n_actions, n_states, epsilon, alpha, gamma)
        self.env = env
        
    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions) # Explore
        else:
            action = np.argmax(self.Q[state]) # Exploit
        
        return action
        
    def update(self, state, action, reward, next_state, next_action): # Augment arguments if necessary
        self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes): 
            self.env.reset()
            state = self.env.state()
            action = self.select_action(state)
            cumulative_reward = 0

            while not self.env.done():
                reward = self.env.step(action)
                next_state = self.env.state()
                next_action = self.select_action(next_state)

                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                cumulative_reward += reward

            episode_returns.append(cumulative_reward)

        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
