import numpy as np 
from game_tournament.game import Player 

class player(Player): 
    
    name = 'xxx' # <--- write your name here!!!

    def play(self, U1, U2): 
        NA1, NA2 = U1.shape
        a = [] # <--- write your code here!!!

        # whatever your player function does, it must return something in {0, 1, ..., NA1-1} 
        assert a in np.arange(NA1), f'Action {a} is not in the range [0, ..., {NA1-1}]'

        return a # function must return an integer
