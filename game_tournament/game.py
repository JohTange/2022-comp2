import numpy as np
import pandas as pd
import os
import sys
import importlib.util
from itertools import combinations, cycle
from tqdm import tqdm as progress_bar

# Author: Anders Munk-Nielsen
#
# Tournament: the top level 
# Game: A tournament consists of a number of games. 
# Rounds: Each game can be played over many rounds. 
#   Payoffs are the discounted sum over the rounds. 
#
# Static vs. repeated games: 
# Player functions can only see the history of the game 
# if it is given to them. Hence, it is the "get_action()" 
# function of the game that determines what players will 
# be able to observe. 
#
# Writing a new game 
# inherit from either continuous or discrete game superclass 
# and write __init__, get_action, and check_action. 
# get_action should ensure that the correct variables are 
# passed to player functions so it needs to be syncronized 
# with how they are written. 

class Player:
    def __init__(self, filepath:str = None):
        self.filepath = filepath # the filename of the player (nice to store for debugging)
        self.i = None
        pass

    def __str__(self):
        return f"Player {self.i}: {self.name}"

    def __hash__(self): 
        return hash(self.name) # player name must be unique 


# A game class for pure strategy games of complete and perfect information

class GeneralGame: 

    def __str__(self): 
        if hasattr(self, 'history') and hasattr(self,'name'): 
            T,N = self.history.shape 
            s = f'{self.name}: played {T} rounds'
        elif hasattr(self, 'name'): 
            s = f'{self.name}'
        else: 
            s = 'Game objects without name or history'
        
        return s

    def compute_payoff_history(self) -> np.ndarray: 
        actions = self.history
        u1,u2 = self.state['payoffs']

        N = len(self.players)
        assert N == 2 , f'Only implemented for two-player games'
        if actions.ndim == 1: 
            aa = actions.reshape(1, N)
        else: 
            aa = actions 
        
        payoffs = np.empty(aa.shape, dtype=float)

        for t in range(aa.shape[0]): 
            a1, a2 = actions[t, :]

            # the game is symmetric 
            payoffs[t,0] = u1(a1, a2)
            payoffs[t,1] = u2(a2, a1) 

        return payoffs 

    def score_game(self, normalization:str = 'mean') -> np.ndarray: 
        '''score_game(): 
        Args
            normalization: 'mean' for the mean over rounds; otherwise use sum.
        Returns: 
            final_payoffs (N,): Discounted average payoffs over the rounds played
        '''
        T,N = self.history.shape
        assert T>0, f'Cannot score a match that has not been played'
        assert normalization in ['mean', 'sum'], f'normalization must be sum or mean.'

        payoffs = self.compute_payoff_history()
        
        # comptue discounted profits: sum_t=1^T disc^t * payoff 
        discounts = self.state['discount_factor'] ** np.arange(T) # (T,)
        final_payoffs = discounts @ payoffs # (T,) @ (T,N) = (N,)
        if (normalization == 'mean'): 
            final_payoffs /= T
        # else: sum, so no changing required

        return final_payoffs 

    def play_game(self, T:int = 1, beta:float = 1.0):
        """play_game: Plays T rounds of 1 vs. 2, then T rounds of 2 vs. 1. 

            Populates self.history

            INPUTS: 
                self: game instance 
                T: no. rounds to play 
                beta: discounting of round payoffs (should be =1.0 for repeated static games)
                    (static games can be repeated to smooth out any randomness)
            RESULTS: 
                self.history: (T,num_players) array of actions (dtype=self.action_dtype)
            
            Note: we cannot return the history as players may rely on it
        """

        # allocate full history: self.history will be growing over time 
        N = len(self.players)
        full_history = np.empty((T, N), dtype=self.action_dtype)
        self.history = np.empty((0,2), dtype=self.action_dtype) # initialize (otherwise, something fails when t=0 and we access full_history[:t,:], which becomes 1-dim)

        # play game T times with default coniguration
        for t in range(T):
            for i,player in enumerate(self.players): 

                # choose action 
                # get_action() takes care of what inputs the player should see
                # (e.g. the history of play, what states, etc.)
                try: 
                    full_history[t,i] = self.get_action(player) 

                except Exception as e: 
                    # frequently, player functions have bugs etc. in them so let us conveniently
                    # inform the 
                    if player.filepath is not None:
                        raise Exception(f'Error from player {player.name} (file "{player.filepath}"). \n\nPrinting original error message: \n\n{e}')
                    else: 
                        raise Exception(f'Error from player {player.name}. \n\nPrinting original error message: \n\n{e}')


                # verify legality of action 
                self.check_action(full_history[t,i], i)
            
            # show players the history up until now
            self.history = full_history[:t, :] 

        # return the full history 
        self.history = full_history

    def get_game_actions_and_payoffs(self) -> tuple:
        '''
        Returns actions, payoffs
        ''' 
        T,N = self.history.shape
        assert T > 0 , f'Game not played yet, cannot print history'

        payoff_history = self.compute_payoff_history()

        # dataframes
        actions = pd.DataFrame({p.name: self.history[:, i]   for i,p in enumerate(self.players)})
        payoffs = pd.DataFrame({p.name: payoff_history[:, i] for i,p in enumerate(self.players)})

        for df,name in zip([actions, payoffs],
                           ['actions', 'payoffs']): 
            df.columns = pd.MultiIndex.from_tuples(zip(cycle([name]), df.columns))

        tab = actions.join(payoffs)
        tab.index.name = 'Round'
        return tab
        

class ContinuousGame(GeneralGame): 

    def check_action(self, a:float, i_player:int) -> bool:
        '''check_action(): verifies whether a given action is legal in the game 
            Uses self.state['actions']
        
        Returns
            I (bool): True if action is legals
        '''
        pmin,pmax = self.state['actions'][i_player]
        p = self.players[i_player]
        err_str = f'Player {p.name} made an illegal action'
        if p.filepath is not None: 
            err_str += ' (file: "{p.filepath}")'
        assert np.isscalar(a), f'{err_str}: non-scalar action'
        assert (a >= pmin) & (a <= pmax), f'{err_str}: price {a} outside permitted range [{pmin}; {pmax}]'
        return True

class RepeatedBertrandGame(ContinuousGame): 
    n = 2 # only for two players 

    def __init__(self, player1, player2, demand_function, marginal_cost, price_range, discount_factor): 
        '''Repeated Bertrand Game 
            INPUTS: 
                player1, player2: player functions, having method play(state, history). 
                demand_function: function handle taking 3 inputs: p1, p2, i
                marginal_cost: scalar, common marginal cost parameter 
                price_range: tuple, (pmin, pmax): action space is [pmin; pmax]
                discount_factor: the amount by which the future is discounted 
                    (this number should be used to compute total payoffs, although that 
                    parameter is called beta elsewhere in the code.)

            [no output, modifies the object self]
        '''
        # checks 
        assert isinstance(price_range, tuple), f'price_range must be a tuple'
        assert len(price_range) == 2, f'Price range must have two elements, (pmin, pmax)'
        assert np.isscalar(marginal_cost), f'marginal_cost must be scalar'
        assert marginal_cost >= 0.0, f'marginal_cost must be non-negative. '

        self.action_dtype = float

        pmin, pmax = price_range
        assert marginal_cost < pmax, f'Marginal cost ({marginal_cost}) must be less than pmax ({pmax})'

        self.players = [player1, player2]
        for i in [0,1]: 
            self.players[i].i = i 

        # very basic checks 
        for player in self.players: 
            hasattr(player, 'name'), f'Player function has no name!'
            hasattr(player, 'play'), f'Player function, {player.name}, has no play() sub-function'
        self.name = f"{self.players[0].name} vs. {self.players[1].name}"
        
        # the state variables that players can see 
        self.state = dict()
        pi1 = lambda p1, p2 : demand_function(p1, p2) * (p1 - marginal_cost)
        self.state['payoffs'] = [pi1, pi1] # the two firms are symmetric, so they face identical demand curves 
        self.state['actions'] = [price_range, price_range]

        # additional relevant information about the game 
        self.state['discount_factor'] = discount_factor
        self.state['marginal_cost'] = marginal_cost 

        self.history = np.empty((0,2), dtype=self.action_dtype) # initialize: shows that no match rounds have been played 
   
    def get_action(self, player): 
        T,N = self.history.shape
        if T == 0: 
            history_own = np.array([]) # empty array 
            history_opponent = np.array([]) # empty array
        else: 
            assert player.i is not None, f'player.i is not assigned'
            history_own = self.history[:, player.i]
            history_opponent = self.history[:, 1-player.i]

        pmin, pmax = self.state['actions'][player.i]
        f_profit = self.state['payoffs'][player.i]
        a = player.play(f_profit, pmin, pmax, history_own, history_opponent,
                        self.state['marginal_cost'], self.state['discount_factor'])
        return a

class DiscreteGame(GeneralGame):
    n = 2  # only for 2-player games

    def __init__(self, player1, player2, U1:np.ndarray, U2:np.ndarray, discount_factor:float = 1.0, action_names = None):
        """Bimatrix game
        player1, player2: player classes, must have method "play()"
        U: payoff matrix for both players. Rows = # of actions of player 1,
                cols = # of actions of players 2
        action_names: [optional] (list of lists). A list of lists
            of names of actions, [iplayer, iaction].
        """
        NA1, NA2 = U1.shape 
        assert U2.shape[0] == NA1 
        assert U2.shape[1] == NA2 

        self.action_dtype = int
        self.players = [player1, player2]
        player1.i = 0
        player2.i = 1
        self.name = f"{self.players[0].name} vs. {self.players[1].name}"


        self.state = dict()

        # state['actions']: action space: in a list 
        A1 = np.arange(NA1)
        A2 = np.arange(NA2)
        self.state['actions'] = [A1,A2]

        # state['U']: Utility matrices, in a list 
        self.state['U'] = [U1, U2]

        # state['payoff']
        def payoff1(a1, a2): 
            na1,na2 = U1.shape 
            assert a1 in np.arange(na1), f'Illegal action {a1=} when U1 is {U1.shape}'
            assert a2 in np.arange(na2), f'Illegal action {a2=} when U1 is {U1.shape}'
            return U1[a1, a2]
        def payoff2(a2, a1): 
            na1, na2 = U2.shape 
            assert a1 in np.arange(na1), f'Illegal action {a1=} when U1 is {U1.shape}'
            assert a2 in np.arange(na2), f'Illegal action {a2=} when U1 is {U1.shape}'
            return U2[a1, a2]
        self.state['payoffs'] = [payoff1, payoff2]

        self.state['discount_factor'] = discount_factor

        if action_names == None:
            action_names = [
                [
                    f"P{player_i+1}A{player_i_action+1}"
                    for player_i_action in range(U1.shape[player_i])
                ]
                for player_i in [0, 1]
            ]
        else:
            assert (
                len(action_names) == 2
            ), f"Must be one list of action names per player"
            assert (
                len(action_names[0]) == U1.shape[0]
            ), f"One name per action (player 1)"
            assert (
                len(action_names[1]) == U1.shape[1]
            ), f"One name per action (player 2: found {len(action_names[1])} but U1.shape[1]={U1.shape[1]})"

        self.state["action_names"] = action_names

        self.history = np.empty((0,2), dtype=self.action_dtype)

    def get_action(self, player): 
        U1,U2 = self.state['U']
        if player.i  == 0:
            u1,u2 = U1,U2
        elif player.i == 1:
            u1 = U2.T
            u2 = U1.T
        else: 
            raise Exception(f'Unexpected player number {player.i}. Only 0 and 1 allowed.')

        a = player.play(u1, u2) # player is not given the history of the game
        return a

    def check_action(self, a:int, i_player:int) -> bool:
        A = self.state['actions'][i_player]
        p = self.players[i_player]
        err_str = f'Player {p.name} made an illegal action'
        assert np.isscalar(a), f'{err_str}: non-scalar action'
        assert a in A, f'{err_str}: action not in action set'
        return True

class Tournament:
    """A game theory tournament.

    Takes a path to modules "players_filepath", e.g. = './players/'
    and a game class "game" as input. Outputs a winner of the tournament.
    """
    def __init__(self, players_filepath:str, Game, game_data:dict, T:int = 1, tournament_name:str = None):

        assert os.path.isdir(players_filepath), f'Input, "players_filepath", must be a directory'
        assert players_filepath.endswith('/'), f'"players_filepath" should end with "/".'

        self.Game = Game # class for the game 
        self.game_data = game_data # common game settings
        self.games = [] # instances of the game (individual matchups): won't be filled out until we run the tournament 
        self.T = T

        self.players = [] # will hold a list of player objects
        for file in os.listdir(players_filepath):
            if file.endswith('.py'):
                self.players.append(load_player_module(players_filepath, file))

        # now we have populated the list self.players
        assert len(self.players) > 0, f'No player.py files found in directory "{players_filepath}"'
        assert len(self.players) > 1, f'Only {len(self.players)} player functions found in "{players_filepath}": need at least 2.'
        self.player_names = [p.name for p in self.players]

        # verify that we have no duplicates 
        c = pd.value_counts(self.player_names) 
        I = c>1 # player names with duplicate entries 
        if I.any(): 
            raise Exception(f'Duplicate player names found for {c[I].values}')

        # give the tournament a name 
        if tournament_name is None: 
            self.name = f'{len(self.players)}-player tournament'
        else: 
            self.name = tournament_name 

    def __str__(self): 
        '''print information about the tournament instance'''
        if len(self.games) > 0: 
            N = len(self.players)
            winners = self.declare_winners()
            win_string = ' and '.join(winners)
            if len(winners) == 1: 
                return f'Tournament winner was: {win_string} (against {N-1} opponents)'
            elif (len(winners) > 1) and (len(winners) < len(self.players)): 
                return f'Tournament draw among {len(winners)} of {len(self.players)} players: {win_string}'
            elif len(winners) == len(self.players): 
                return f'Tournament ended in a draw among all {N} players: {win_string}'
            else: 
                raise Exception(f'Unexpected return from declare_winners(): {winners}')
        elif len(self.players) > 0: 
            return f'Tournament ready with {len(self.players)} players'
        else: 
            return f'Tournament not fully initialized.'

    def all_play_all(self):
        '''all_play_all: 
            Loop through all combinations of players and make them fight. 
            Results: 
                self.games: list of games with their results 
        '''
        player_pairs = combinations(self.players, 2)
        
        for player_i, player_j in progress_bar(player_pairs): 

            this_game = self.Game(player_i, player_j, **self.game_data) # initiate instance of game
            this_game.play_game(self.T)
            this_game.points = this_game.score_game()
            self.games.append(this_game)

    def get_matchup_results(self) -> pd.core.frame.DataFrame: 
                    
        # create an empty N*N dataframe 
        match_results = pd.DataFrame(index=self.player_names, columns=self.player_names, dtype=float)

        for g in self.games: 
            match_results.loc[g.players[0].name, g.players[1].name] = g.points[0]
            match_results.loc[g.players[1].name, g.players[0].name] = g.points[1]
        match_results
        match_results.columns.name = 'Opponent'
        match_results.index.name = 'Player'
        return match_results

    def scoreboard(self) -> pd.core.frame.DataFrame: 
        match_results = self.get_matchup_results()
        return match_results.mean(axis=1).sort_values(ascending=False).to_frame(self.name)

    def declare_winners(self) -> list:
        '''
        Returns
            list of string(s): names of the winner(s)
        '''
        N = len(self.players)
        points = self.scoreboard().iloc[:, 0] # there is only one column
        assert points.shape[0] == N, f'Unexected size of scoreboard {points.shape[0]}, but there are {N} players'
        I = points == points.max()
        winners = points.loc[I].index.values
        return winners

    def run(self) -> pd.core.frame.DataFrame(): 
        self.all_play_all()
        print(self)
        return self.scoreboard()

    def get_payoffs(self, i_game): 
        '''get_payoffs(): 
        Returns
            list of dataframes for each game played
        '''
        assert len(self.games) > 0, f'No games have been played yet'
        assert len(self.games) >= i_game+1, f'Only {len(self.games)} games recorded, {i_game=} requested.'
        g = self.games[i_game]
        p = pd.DataFrame(g.compute_payoff_history(), 
            columns=[player.name for player in g.players]
        )
        p.index.name = 'Round'
        
        return p

    def get_actions(self, i_game): 
        g = self.games[i_game]
        pnames = [p.name for p in g.players]
        a = pd.DataFrame(g.history, columns=pnames)
        a.index.name = 'Round'
        return a

    def plot_results(self): 
        '''plot_results
        Returns
            matplotlib axis
        '''
        m = self.get_game_results()
        ax = m.plot.bar()
        return ax


def load_player_module(path:str, file:str):
    '''load_player_module: loads a single player module from the given file 
        Args: 
            player_file: string, filename including the path 

        Returns: 
            player class instance 
    ''' 
    assert file.endswith('.py'), 'Only files ending with ".py" can be loaded'
    player_file = path + file 
    mod_name = file[:-3]
    try: 
        spec = importlib.util.spec_from_file_location(mod_name,player_file)
        player_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = player_module
        spec.loader.exec_module(player_module)
        p = player_module.player(player_file) 
        # the player instance is created here
    except: 
        raise Exception(f'Failed to read player file {player_file}')
        
    return p

