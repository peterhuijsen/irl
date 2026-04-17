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
        
    def policy(self, state):
        best_action = np.argmax(self.Q[state])
        action_probabilities = np.ones(self.n_actions) * self.epsilon / self.n_actions
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities
        
    def select_action(self, state):
        action_probabilities = self.policy(state)
        action = np.random.choice(self.n_actions, p=action_probabilities)
        return action
     
    def train(self, n_episodes):
        raise NotImplementedError("This method should be implemented by subclasses of Agent.")
        

class QLearningAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
        
    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule
        if done:
            self.Q[state, action] = self.Q[state, action] + self.alpha * \
                (reward - self.Q[state, action])    
            return
        
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
        
    def train(self, n_episodes):
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0
            
            state = self.env.state()

            while not self.env.done():
                action = self.select_action(state)
                reward = self.env.step(action)
                next_state = self.env.state()
                
                self.update(state, action, reward, next_state, self.env.done())

                state = next_state
                cumulative_reward += reward

            episode_returns[episode] = cumulative_reward

        return episode_returns


class SARSAAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
          
    def update(self, state, action, reward, next_state, next_action, done):
        # SARSA update rule
        if done:
            self.Q[state, action] = self.Q[state, action] + self.alpha * \
                (reward - self.Q[state, action])
            return
        
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
        
    def train(self, n_episodes):
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0

            state = self.env.state()
            action = self.select_action(state)

            while not self.env.done():
                reward = self.env.step(action)

                next_state = self.env.state()
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, self.env.done())

                state, action = next_state, next_action
                cumulative_reward += reward

            episode_returns[episode] = cumulative_reward

        return episode_returns


class ExpectedSARSAAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
        
    def update(self, state, action, reward, next_state, done):
        # Expected SARSA update rule
        if done:
            self.Q[state, action] = self.Q[state, action] + self.alpha * \
                (reward - self.Q[state, action])
            return
        
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * np.sum(self.Q[next_state] * self.policy(next_state)) - self.Q[state, action])

    def train(self, n_episodes):
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0
            
            state = self.env.state()

            while not self.env.done():
                action = self.select_action(state)
                reward = self.env.step(action)
                next_state = self.env.state()
                
                self.update(state, action, reward, next_state, self.env.done())

                state = next_state
                cumulative_reward += reward

            episode_returns[episode] = cumulative_reward

        return episode_returns   

class DecayingExpectedSARSAAgent(ExpectedSARSAAgent):
    def __init__(self, env: Environment, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0, decay_rate=0.99, decay_floor=0.01):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
        self.decay_rate = decay_rate
        self.decay_floor = decay_floor

    def train(self, n_episodes):
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0
            
            state = self.env.state()

            while not self.env.done():
                action = self.select_action(state)
                reward = self.env.step(action)
                next_state = self.env.state()
                
                self.update(state, action, reward, next_state, self.env.done())

                state = next_state
                cumulative_reward += reward

            episode_returns[episode] = cumulative_reward

            # Decrease epsilon after each episode to encourage more exploitation over time.
            self.epsilon = max(self.decay_floor, self.epsilon * self.decay_rate)

        return episode_returns


class nStepSARSAAgent(Agent):
    def __init__(self, env: Environment, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        super().__init__(env, n_actions, n_states, epsilon, alpha, gamma)
        self.n = n

    def update_step(self, state, action, rewards, next_state, next_action):
        # If we are not at a terminal node, then we only update the state-action pair
        # which is n steps back.
        bootstrap = self.gamma**len(rewards) * self.Q[next_state, next_action] if not self.env.done() else 0
        G = sum([self.gamma**i * rewards[i] for i in range(len(rewards))]) + bootstrap
        self.Q[state, action] = self.Q[state, action] + self.alpha * (
            G - self.Q[state, action]
        )
        
    def update_flush(self, states, actions, rewards):
        # If we are at a terminal node, then we update all the state-action pairs 
        # which still need to be updated.
        for i in range(len(states)):
            # Since we now know the final results we do not need to bootstrap, and 
            # we can calculate the return G for each state-action pair directly 
            # from the rewards received after that state-action pair.
            G = sum([self.gamma**j * rewards[i + j] for j in range(len(rewards) - i)])
            self.Q[states[i], actions[i]] = self.Q[states[i], actions[i]] + self.alpha * (
                G - self.Q[states[i], actions[i]]
            )
        
    def train(self, n_episodes):
        episode_returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0
            states, actions, rewards = [], [], []

            state = self.env.state()
            action = self.select_action(state)

            while not self.env.done():                
                reward = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                next_state = self.env.state()
                next_action = self.select_action(next_state)

                # We skip the first n steps before updating.
                if len(actions) >= self.n:
                    self.update_step(
                        state=states[-self.n],
                        action=actions[-self.n],
                        rewards=rewards[-self.n:],
                        next_state=next_state,
                        next_action=next_action
                    )
                    
                state, action = next_state, next_action
                cumulative_reward += reward

            # If we have not yet updated the last n state-action pairs, then 
            # we need to update them now with the rewards received until the 
            # end of the episode.
            tail = min(len(actions), self.n - 1)
            if tail > 0:
                self.update_flush(
                    states=states[-tail:], 
                    actions=actions[-tail:], 
                    rewards=rewards[-tail:]
                )

            episode_returns[episode] = cumulative_reward

        return episode_returns  
    