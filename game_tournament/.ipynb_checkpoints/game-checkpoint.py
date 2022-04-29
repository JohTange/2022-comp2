import numpy as np
import os
import importlib.util
import sys
from itertools import combinations, groupby
import tqdm
import pandas as pd



class Player:
    def __init__(self, name):
        self.name = name  # used for printing
        self.i = np.nan  # player number, not assigned yet

    def __str__(self):
        return f"Player {self.i}: {self.name}"

    def play(self, state):
        # fill this out
        pass


# A game class for pure strategy games of complete and perfect information


class DiscreteGame:
    n = 2  # only for 2-player games

    def __init__(self, player1, player2, U1, U2, action_names=[]):
        """Bimatrix game
        player1, player2: player classes, must have method "play()"
        U1, U2: payoff matrices. Rows = # of actions of player 1,
                cols = # of actions of players 2
        action_names: [optional] (list of lists). A list of lists
            of names of actions, [iplayer, iaction].
        """

        assert U1.shape == U2.shape
        n_actions_player1, n_actions_player2 = U1.shape

        self.players = [player1, player2]

        self.name = f"{self.players[0].name} vs. {self.players[1].name}"

        # assign player number
        for i in [0, 1]:
            self.players[i].i = i

        self.state = dict()
        self.state["payoffs"] = [U1, U2]

        if action_names == []:
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

        self.state["actions"] = action_names

        self.history = []

    def flip_player_roles(self):
        """flip_player_roles: Makes the previous player 1 into player 2 and
        vice versa. Also resets the history of the game.
        """
        self.players.reverse()

        # update the player numbers
        for i in range(self.n):
            self.players[i].i = i

        self.name = f"{self.players[0].name} vs. {self.players[1].name}"

        # reset history
        self.history = []

    def payoffs(self, actions):

        pay = np.empty((self.n,))
        assert len(actions) == self.n

        actions_player1 = actions[0]
        actions_player2 = actions[1]

        for i in range(self.n):
            self.check_action(actions[i], i)
            U = self.state["payoffs"][i]
            pay[i] = U[actions_player1, actions_player2]

        return pay

    def check_action(self, action, i):
        U = self.state["payoffs"][i]
        n_actions_player_i = U.shape[i]

        if action in range(n_actions_player_i):
            return True
        else:
            return False

    def play_round(self, DOPRINT=False):
        """play_round: plays a single round of the game, storing actions in the history"""
        actions_played = np.zeros((self.n,), dtype="int")
        for i in range(self.n):
            action_index = self.players[i].play(self.state)
            if not self.check_action(action_index, i):
                j = 1 - i
                raise Exception(
                    f"{self.players[i].name} did something illegal (action_index={action_index}) and is disqualified! {self.players[j].name} wins!"
                )
            else:
                actions_played[i] = action_index

        if DOPRINT:
            u = self.payoffs(actions_played)
            for i in range(self.n):
                a_ = self.state["actions"][i][actions_played[i]]
                print(f"{self.players[i].name} played {a_} getting {u[i]}")

        self.history.append(actions_played)

    def compute_total_payoff_from_history(self, beta=1.0):
        """compute_total_payoff_from_history: uses the history of
        the game and computes total discounted sum of winnings
        """
        T = len(self.history)
        assert T > 0, f"History is empty!"
        payoffs = np.empty((T, self.n))
        for t, actions_played in enumerate(self.history):
            payoffs[t, :] = self.payoffs(actions_played) * beta ** t

        tot_winnings = payoffs.sum(0)

        return tot_winnings

    def declare_winner(self, T=10):
        """This might not make sense in a matrix game"""

        # reset any history
        self.history = []

        # play game T times with default coniguration
        for t in range(T):
            self.play_round()
        winnings1 = self.compute_total_payoff_from_history(beta=1.0)

        # flip roles of players and do the same
        self.flip_player_roles()

        for t in range(T):
            self.play_round()
        winnings2 = self.compute_total_payoff_from_history(beta=1.0)

        self.flip_player_roles()
        winnings2 = np.flip(winnings2)

        self.tot_winnings = winnings1 + winnings2

        # determine winner or if draw
        if self.tot_winnings[0] > self.tot_winnings[1]:
            # print(f"{self.players[0].name} won!")
            self.subgame_points = [self.players[0].name, self.players[0].name]
        elif self.tot_winnings[0] < self.tot_winnings[1]:
            # print(f"{self.players[1].name} won!")
            self.subgame_points = [self.players[1].name, self.players[1].name]
        elif self.tot_winnings[0] == self.tot_winnings[1]:
            # print(f"Draw in {self.name}!")
            self.subgame_points = [self.players[0].name, self.players[1].name]
        else:
            # this should never happen! means the game has an error somehow
            raise Exception(
                f"Unexpected outcome for total winnings: {self.tot_winnings}! Maybe NaNs?"
            )

class Tournament:
    """A game theory tournament.

    Takes a path to modules "players_filepath" 
    and a game class "game" as input. Outputs a winner of the tournament.
    """
    def __init__(self, players_filepath, game):
        self.players_filepath = players_filepath
        self.game = game
        self.player_files = []
        for file in os.listdir(self.players_filepath):
            if file[-3:] == ".py":
                self.player_files.append(file)
        self.num_players = len(self.player_files)
        # player file index
        self.player1_index = 0
        self.player2_index = 1
        # player files
        self.player1_file = None
        self.player2_file = None
        # player modules
        self.player1 = None
        self.player2 = None
        # winner of latest game in tournament
        self.game_winner = None
        self.tournament_history = dict()
        # winner of tournament
        self.tournament_winner = None
        self.tournament_rank = None

    def __str__(self): 
        if self.tournament_rank is not None: 
            return f'Finished tournament, winner was: {self.tournament_rank.Name[0]}'
        elif self.num_players is not None: 
            return f'Tournament ready with {self.num_players} players'
        else: 
            return f'Tournament not fully initialized. '

    def load_player_modules(self, player_file):
        spec = importlib.util.spec_from_file_location(
        "module.name",
        os.path.join(self.players_filepath, player_file),
        )
        player_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = player_module
        spec.loader.exec_module(player_module)
        return player_module.player()


    def update_players(self):
        """Updates the players based on result of previous game"""
        self.player1 = self.load_player_modules(self.player1_file)
        self.player2 = self.load_player_modules(self.player2_file)

    def all_play_all(self):
        self.tournament_history["all_play_all_results"] = []
        self.matches = combinations(self.player_files, 2)
        # Making sure there are not any player files with the same names.
        assert sum([i==j for i, j in self.matches]) == 0, f"Duplicate players"
        self.matches = combinations(self.player_files, 2)
        for player_i, player_j in tqdm.tqdm(list(self.matches)):
            self.player1_file = player_i
            self.player2_file = player_j
            self.update_players()
            self.game_played = self.game(self.player1, self.player2, U1=self.U1, U2=self.U2, action_names=self.action_names)
            self.game_played.declare_winner()
            self.tournament_history["all_play_all_results"].extend(self.game_played.subgame_points)


    def calculate_wins(self):
        points_ = dict()
        points_["Name"] = list()
        points_["Points"] = list()
        for i in set(self.tournament_history["all_play_all_results"]):
            points_["Name"].append(i)
            points_["Points"].append(self.tournament_history["all_play_all_results"].count(i))
        _df_points = pd.DataFrame(points_)

        self.tournament_rank = _df_points.sort_values(by=["Points", "Name"], ascending=False)
        pass

    def start_tournament(self, U1, U2, action_names=[]):
        """assigns self.tournament_rank, a dataframe with the points for each player Name 
        """
        self.U1 = U1
        self.U2 = U2
        self.action_names = action_names
        self.all_play_all()
        self.calculate_wins()
        print("\nTop placements are:\n", self.tournament_rank.head(), sep="")
        
 
