import numpy as np 
from game_tournament.game import Player 

class player(Player): 
    
    name = 'Randawg'

    def play(self, U1, U2): 
        NA1, NA2 = U1.shape 
        A1 = np.arange(NA1)

        a = np.random.choice(A1)
        return a 
