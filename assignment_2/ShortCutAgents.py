import numpy as np

from ShortCutEnvironment import Environment

class Agent(object):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.env = env

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
     
    def train(self, n_episodes):
        raise NotImplementedError("This method should be implemented by subclasses of Agent.")
        

class QLearningAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
        
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
        
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            state = self.env.state()
            cumulative_reward = 0

            while not self.env.done():
                # Calculate the action and reward for that action.
                action = self.select_action(state)
                reward = self.env.step(action)

                # Update the Q-values based on the transition.
                next_state = self.env.state()
                self.update(state, action, reward, next_state)

                state = next_state
                cumulative_reward += reward

            episode_returns[episode] = cumulative_reward

        return episode_returns


class SARSAAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
          
    def update(self, state, action, reward, next_state, next_action):
        # SARSA update rule
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
        
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
                # First, calculate the reward.
                reward = self.env.step(action)

                # Calculate the next state, action, and reward.
                next_state = self.env.state()
                next_action = self.select_action(next_state)

                # Update the Q-values based on the transition.
                self.update(state, action, reward, next_state, next_action)

                # Update next states
                state, action = next_state, next_action
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
    
    
