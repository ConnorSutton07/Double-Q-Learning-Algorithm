"""
Defines grid that makes up example enviornment
for training a Deep-Q Network

"""

class Grid:
    def __init__(self) -> None:
        self.state = None

    def reset(self) -> None:
        self.state = (2, 0)
        return self.state 
    
    def step(self, action: tuple) -> tuple:
        row_step = action[0]
        col_step = action[1]
        row = max(0, min(2, self.state[0] + row_step))
        col = max(0, min(2, self.state[1] + col_step))

        terminal = False
        reward = -1

        if (row, col) == (0, 2): # goal state
            reward = 0
            terminal = True

        self.state = (row, col)
        return (row, col), reward, terminal 

    
        