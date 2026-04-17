import numpy as np
from ShortCutAgents import SARSAAgent
from ShortCutEnvironment import WindyShortcutEnvironment

env = WindyShortcutEnvironment()
agent = SARSAAgent(n_actions=4, n_states=12**2, epsilon=0.1, alpha=0.1, gamma=1.0, env=env)
agent.train(n_episodes=10000)

actions = ['^', 'v', '<', '>']
Q = agent.Q

display = []
for y in range(12):
    row = []
    for x in range(12):
        if env.s[y, x] in ['C', 'G']:
            row.append(env.s[y, x])
            continue
        best_a = np.argmax(Q[y*12+x])
        if np.max(Q[y*12+x]) == 0 and np.min(Q[y*12+x]) == 0:
            row.append('0')
        else:
            row.append(actions[best_a])
    display.append(" ".join(row))

print("\n".join(display))
