import numpy as np 
from game_tournament.game import Player 

class player(Player): 
    
    name = '1st order'

    def play(self, U1, U2): 
        NA1, NA2 = U1.shape 
        A1 = np.arange(NA1)

        alpha_2 = 1/NA2 * np.ones((NA2,))
        Eu1 = U1 @ alpha_2 
        a = Eu1.argmax()

        if not (a in A1): # this should never happen 
            a = np.random.choice(A1)

        return a 
