#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.Q_sa = np.zeros((n_states, n_actions), dtype=float)
        self.N = np.zeros((n_states, n_actions, n_states), dtype=int)
        self.R = np.zeros((n_states, n_actions, n_states), dtype=float)

        self.p = np.zeros((n_states, n_actions, n_states), dtype=float)
        self.r = np.zeros((n_states, n_actions, n_states), dtype=float)
        
    def policy(self, state, epsilon) -> np.ndarray:
        best_action = np.argmax(self.Q_sa[state])
        action_probabilities = np.ones(self.n_actions) * epsilon / self.n_actions
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities
        
    def select_action(self, state, epsilon) -> int:
        action_probabilities = self.policy(state, epsilon)
        action = np.random.choice(self.n_actions, p=action_probabilities)
        return action

    def update(self,s,a,r,done,s_next,n_planning_updates):
        def update_model():
            self.N[s, a, s_next] += 1
            self.R[s, a, s_next] += r

            self.p[s, a] = self.N[s, a] / np.sum(self.N[s, a])
            self.r[s, a, s_next] = self.R[s, a, s_next] / self.N[s, a, s_next]

        def update_Q_sa(s, a, r, s_next, done):
            self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (r + self.gamma * (not done) * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])

        # Update the model based on the observed transition
        update_model()

        # Update the Q-values based on the observed transition
        update_Q_sa(s, a, r, s_next, done)

        # Planning updates
        for _ in range(n_planning_updates):
            if np.sum(self.N, axis=(0, 1, 2)) <= 0:
                return

            # Get a random already visited and performed state and action.
            s = np.random.choice(
                np.where(np.sum(self.N, axis=(1, 2)) > 0)[0]
            )
            a = np.random.choice(
                np.where(np.sum(self.N[s], axis=1) > 0)[0]
            )

            # Then sample the next action and get the reward from that action
            s_next = np.random.choice(range(self.n_states), p=self.p[s, a, :])
            r = self.r[s, a, s_next]

            # Update the Q-values based on the model
            update_Q_sa(s, a, r, s_next, done=False)

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        
        self.Q_sa = np.zeros((n_states, n_actions), dtype=float)
        self.N = np.zeros((n_states, n_actions, n_states), dtype=int)
        self.R = np.zeros((n_states, n_actions, n_states), dtype=float)

        self.p = np.zeros((n_states, n_actions, n_states), dtype=float)
        self.r = np.zeros((n_states, n_actions, n_states), dtype=float)
        
    def policy(self, state, epsilon) -> np.ndarray:
        best_action = np.argmax(self.Q_sa[state])
        action_probabilities = np.ones(self.n_actions) * epsilon / self.n_actions
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities
        
    def select_action(self, state, epsilon) -> int:
        action_probabilities = self.policy(state, epsilon)
        action = np.random.choice(self.n_actions, p=action_probabilities)
        return action
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        def update_model():
            self.N[s, a, s_next] += 1
            self.R[s, a, s_next] += r

            self.p[s, a] = self.N[s, a] / np.sum(self.N[s, a])
            self.r[s, a, s_next] = self.R[s, a, s_next] / self.N[s, a, s_next]

        def update_queue(s, a, r, s_next, done):
            p = np.abs(r + self.gamma * (not done) * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])
            if p > self.priority_cutoff:
                self.queue.put((-p, (s, a)))   

        def update_Q_sa(s, a, r, s_next, done):
            self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (r + self.gamma * (not done) * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])

        # Update the model based on the observed transition
        update_model()

        # Update the PQ
        update_queue(s, a, r, s_next, done)

        # Planning updates
        for _ in range(n_planning_updates):
            if self.queue.empty():
                return

            # Get the state and action from the PQ
            _, (s, a) = self.queue.get()
    
            # Then sample the next action and get the reward from that action
            s_next = np.random.choice(range(self.n_states), p=self.p[s, a, :])
            r = self.r[s, a, s_next]

            # Update the Q-values based on the model
            update_Q_sa(s, a, r, s_next, done=False)

            # Then we go through all the ways we could have ended up at the current
            # state and add those to the queue
            possibilities = list(zip(*np.where(self.N[..., s] > 0)))
            for s_p, a_p in possibilities:
                r_p = self.r[s_p, a_p, s]
                update_queue(s_p, a_p, r_p, s, done=False)

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = False
    plot_optimal_policy = True
    step_pause = 0.000000001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):      
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next

    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy, step_pause=0)
            
    
if __name__ == '__main__':
    test()
