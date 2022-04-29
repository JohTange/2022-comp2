import numpy as np 
from game_tournament.game import Player 

class player(Player): 
    
    name = '2nd order'

    def play(self, U1, U2): 
        NA1, NA2 = U1.shape 
        A1 = np.arange(NA1)

        Eu2 = U2.mean(axis=0)
        a2 = Eu2.argmax()

        Eu1 = U1[:, a2]
        a1 = Eu1.argmax()

        if a1 not in A1: 
            a1 = np.random.choice(A1)

        return a1 
