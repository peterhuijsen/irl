import random
import numpy as np

class Environment(object):

    def __init__(self):
        pass

    def reset(self):
        '''Reset the environment.
        
        Returns:
           starting_position: Starting position of the agent.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def render(self):
        '''Render environment to screen.'''
        raise Exception("Must be implemented by subclass.")

    def render_greedy(self, Q):
        '''Render the greedy policy based on the Q-array to screen.'''
        raise Exception("Must be implemented by subclass.")
    
    def step(self, action):
        '''Take action.
        
        Arguments:
           action: action to take.
        
        Returns:
           reward: reward of action taken.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def possible_actions(self):
        '''Return list of possible actions in current state.
        
        Returns:
          actions: list of possible actions.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def state(self):
        '''Return current state.

        Returns:
          state: environment-specific representation of current state.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def state_size(self):
        '''Return the number of elements of the state space.

        Returns:
          state_size: number of elements of the state space.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def action_size(self):
        '''Return the number of elements of the action space.

        Returns:
          state_size: number of elements of the action space.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def done(self):
        '''Return whether current episode is finished and environment should be reset.

        Returns:
          done: True if current episode is finished.
        '''
        raise Exception("Must be implemented by subclass.")

class ShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[self.r//3:, self.c//3:2*self.c//3] = 'C'
        s[5*self.r//6-1,:self.c//2] = 'X'
        s[2*self.r//3:5*self.r//6:,self.c//2] = 'X'
        s[2*self.r//3,self.c//2:2*self.c//3] = 'X'
        s[2*self.r//3, 2*self.c//3] = 'G'
        self.s = s
        self.reset()
    
    def reset(self):
        self.x = self.c//6
        rand_number = int(2*self.rng.random())
        if rand_number:
            self.y = 5*self.r//6 - 1
        else:
            self.y = self.r//6
        self.starty = self.y
        self.isdone = False
        return rand_number
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        
        if action == 0:
            if self.y>0:
                self.y -= 1
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1
        elif action == 2:
            if self.x>0:
                self.x -= 1
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.x = self.c//6
                self.y = self.starty
                return -100
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))

    def render_greedy(self, Q):
        """Render environment with Q-table policy visualization.
    
        Args:
            Q: Q-table with shape [state_size, action_size]
        """
        display_grid = np.full((self.r, self.c), " ", dtype=object)
        arrow_map = ["↑", "↓", "←", "→"]
    
        for y in range(self.r):
            for x in range(self.c):
                if self.s[y, x] in ["X", "G", "C"]:
                    display_grid[y, x] = self.s[y, x]
    
        q_actions = np.argmax(Q, axis=1).reshape((self.r, self.c))
        q_max_values = np.max(Q, axis=1).reshape((self.r, self.c))
    
        for y in range(self.r):
            for x in range(self.c):
                if q_max_values[y, x] == 0:
                    display_grid[y, x] = "0"  # Mark unvisited states
                else:
                    display_grid[y, x] = arrow_map[q_actions[y, x]]
    
        display_grid[2 * self.r // 3, 2 * self.c // 3] = "\033[92mG\033[0m"
        display_grid[2, 2] = f"\033[94m{display_grid[2, 2]}\033[0m"
        display_grid[9, 2] = f"\033[94m{display_grid[9, 2]}\033[0m"
    
        display_grid[np.where(self.s == "C")] = "\033[91mC\033[0m"
    
        grid_str = ""
        for row in display_grid:
            grid_str += " ".join(row) + "\n"
        
        print(grid_str)

    def render_greedy_simple(self,Q):
        greedy_actions = np.argmax(Q, 1).reshape((12,12))
        print_string = np.zeros((12, 12), dtype=str)
        print_string[greedy_actions==0] = '^'
        print_string[greedy_actions==1] = 'v'
        print_string[greedy_actions==2] = '<'
        print_string[greedy_actions==3] = '>'
        print_string[np.max(Q, 1).reshape((12, 12))==0] = '0'
        line_breaks = np.zeros((12,1), dtype=str)
        line_breaks[:] = '\n'
        print_string = np.hstack((print_string, line_breaks))
        print(print_string.tobytes().decode('utf-8'))
            

class WindyShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[self.r//3:, self.c//3:2*self.c//3] = 'C'
        s[5*self.r//6-1,:self.c//2] = 'X'
        s[2*self.r//3:5*self.r//6:,self.c//2] = 'X'
        s[2*self.r//3,self.c//2:2*self.c//3] = 'X'
        s[2*self.r//3, 2*self.c//3] = 'G'
        self.s = s
        self.reset()
    
    def reset(self):
        self.x = self.c//6
        rand_number = int(2*self.rng.random())
        if rand_number:
            self.y = 5*self.r//6 - 1
        else:
            self.y = self.r//6
        self.starty = self.y
        self.isdone = False
        return rand_number
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        
        if action == 0:
            if self.y>0:
                self.y -= 1
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1
        elif action == 2:
            if self.x>0:
                self.x -= 1
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1
        
        if self.rng.random()<0.5:
            # Wind!
            if self.y < self.r-1:
                self.y += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.x = self.c//6
                self.y = self.starty
                return -100
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))\

    def render_greedy(self, Q):
        """Render environment with Q-table policy visualization.
    
        Args:
            Q: Q-table with shape [state_size, action_size]
        """
        display_grid = np.full((self.r, self.c), " ", dtype=object)
        arrow_map = ["↑", "↓", "←", "→"]
    
        for y in range(self.r):
            for x in range(self.c):
                if self.s[y, x] in ["X", "G", "C"]:
                    display_grid[y, x] = self.s[y, x]
    
        q_actions = np.argmax(Q, axis=1).reshape((self.r, self.c))
        q_max_values = np.max(Q, axis=1).reshape((self.r, self.c))
    
        for y in range(self.r):
            for x in range(self.c):
                if q_max_values[y, x] == 0:
                    display_grid[y, x] = "0"  # Mark unvisited states
                else:
                    display_grid[y, x] = arrow_map[q_actions[y, x]]
    
        display_grid[2 * self.r // 3, 2 * self.c // 3] = "\033[92mG\033[0m"
        display_grid[2, 2] = f"\033[94m{display_grid[2, 2]}\033[0m"
        display_grid[9, 2] = f"\033[94m{display_grid[9, 2]}\033[0m"
    
        display_grid[np.where(self.s == "C")] = "\033[91mC\033[0m"
    
        grid_str = ""
        for row in display_grid:
            grid_str += " ".join(row) + "\n"
        
        print(grid_str)

    def render_greedy_simple(self,Q):
        greedy_actions = np.argmax(Q, 1).reshape((12,12))
        print_string = np.zeros((12, 12), dtype=str)
        print_string[greedy_actions==0] = '^'
        print_string[greedy_actions==1] = 'v'
        print_string[greedy_actions==2] = '<'
        print_string[greedy_actions==3] = '>'
        print_string[np.max(Q, 1).reshape((12, 12))==0] = '0'
        line_breaks = np.zeros((12,1), dtype=str)
        line_breaks[:] = '\n'
        print_string = np.hstack((print_string, line_breaks))
        print(print_string.tobytes().decode('utf-8'))
        
if __name__ == "__main__":
    # Initialize environment and render it
    env = ShortcutEnvironment()
    env.reset()
    env.render()
    
    # Render greedy policy    
    Q = np.zeros([env.state_size(),env.action_size()])    
    env.render_greedy_simple(Q) # Old version
    env.render_greedy(Q) # Nicer lay-out for your report
    