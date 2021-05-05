import numpy as np
import abc
import util
import math
from game import Agent, Action

MAX_PLAYER = EMPTY = 0
MIN_PLAYER = 1
EMPTY_TILES_WEIGHT = 200
NUM_PAIRS_WEIGHT = 0.5
MAX_TILE_WEIGHT = 10


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # Useful information you can extract from a GameState (game_state.py)
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile

        # Maximize number of tiles that have a large value.
        tiles_score = np.sum((board > np.sqrt(max_tile)))

        # Maximize number of empty tiles.
        empty_tiles_score = np.count_nonzero(board == EMPTY) ** 2

        # Attempt to aggregate high values in one corner.
        corner_score = (board[0, 0] + (board[0, 1] + board[1, 0]) / 32 + (board[1, 1] / 64)) ** 2

        return (corner_score + empty_tiles_score + tiles_score) / 3


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        max_score = -math.inf
        max_action = Action.STOP
        new_depth = self.depth * 2

        # Choose best move according to minmax.
        legal_moves = game_state.get_legal_actions(MAX_PLAYER)
        for action in legal_moves:
            successor = game_state.generate_successor(MAX_PLAYER, action)
            score = self.__minmax_decision(successor, MIN_PLAYER, new_depth - 1)
            if score > max_score:
                max_score, max_action = score, action

        return max_action

    def __minmax_decision(self, game_state, player, cur_depth):
        if cur_depth == 0 or game_state.done:
            return self.evaluation_function(game_state)

        next_player = 1 - player
        if player == MAX_PLAYER:
            legal_moves = game_state.get_agent_legal_actions()
            if not legal_moves:
                return -math.inf
            scores = (
                self.__minmax_decision(game_state.generate_successor(player, action), next_player, cur_depth - 1)
                for action in legal_moves)
            best_score = max(scores)
        else:
            legal_moves = game_state.get_opponent_legal_actions()
            if not legal_moves:
                return math.inf
            scores = (
                self.__minmax_decision(game_state.generate_successor(player, action), next_player, cur_depth - 1)
                for action in legal_moves)
            best_score = min(scores)

        return best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # Recursive helper function for alpha-beta pruning. (max move)
    def __alpha_beta_max_helper(self, state, depth, alpha, beta):
        if depth == 0 or state.done:
            return self.evaluation_function(state)

        legal_moves = state.get_legal_actions(MAX_PLAYER)
        for action in legal_moves:
            successor = state.generate_successor(MAX_PLAYER, action)
            alpha = max(alpha, self.__alpha_beta_min_helper(successor, depth - 1, alpha, beta))
            if beta <= alpha:
                break
        return alpha

    # Recursive helper function for alpha-beta pruning. (min move)
    def __alpha_beta_min_helper(self, state, depth, alpha, beta):
        if depth == 0 or state.done:
            return self.evaluation_function(state)

        legal_moves = state.get_legal_actions(MIN_PLAYER)
        for action in legal_moves:
            successor = state.generate_successor(MIN_PLAYER, action)
            beta = min(beta, self.__alpha_beta_max_helper(successor, depth - 1, alpha, beta))
            if beta <= alpha:
                break
        return beta

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        max_score = -math.inf
        max_action = Action.STOP
        new_depth = self.depth * 2

        # Choose best move according to alpha-beta pruning.
        legal_moves = game_state.get_legal_actions(MAX_PLAYER)
        for action in legal_moves:
            successor = game_state.generate_successor(MAX_PLAYER, action)
            score = self.__alpha_beta_min_helper(successor, new_depth - 1, -math.inf, math.inf)
            if score > max_score:
                max_score, max_action = score, action

        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        max_score = -math.inf
        max_action = Action.STOP
        new_depth = self.depth * 2

        legal_moves = game_state.get_agent_legal_actions()
        for action in legal_moves:
            successor = game_state.generate_successor(MAX_PLAYER, action)
            score = self.__expectimax_decision(successor, MIN_PLAYER, new_depth - 1)
            if score > max_score:
                max_score, max_action = score, action

        return max_action

    def __expectimax_decision(self, game_state, player, cur_depth):
        if cur_depth == 0 or game_state.done:
            return self.evaluation_function(game_state)

        next_player = 1 - player
        if player == MAX_PLAYER:
            legal_moves = game_state.get_agent_legal_actions()
            if not legal_moves:
                return -math.inf
            scores = (
                self.__expectimax_decision(game_state.generate_successor(player, action), next_player, cur_depth - 1)
                for action in legal_moves)
            best_score = max(scores)
        else:
            legal_moves = game_state.get_opponent_legal_actions()
            if not legal_moves:
                return math.inf
            scores = (
                self.__expectimax_decision(game_state.generate_successor(player, action), next_player, cur_depth - 1)
                for action in legal_moves)
            best_score = sum(scores) / len(legal_moves)

        return best_score


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:
    This function evaluates a state using a weighted sum of three features:
    1. The number of empty tiles on the current board.
    2. The number of possible pairs that can be merged on the board- adjacent same numbered pairs are counted
        (diagonal pairs aren't counted since they can't be merged) and a higher weight in the sum is given to
        larger numbers pairs.
    3. The value of the max numbered tile on the board.
    The higher the score of these features the better the state.
    """
    max_tile = current_game_state.max_tile
    board = current_game_state.board

    # Maximize number of empty tiles.
    empty_tiles_score = np.count_nonzero(board == EMPTY)

    # View one tile shifted versions of the board and cuts of the board that we can compare to.
    shift_left = board[:, 1:]
    cut_left = board[:, :-1]
    shift_down = board[1:, :]
    cut_down = board[:-1, :]

    # Maximize number of pairs that can be combined. (Bigger value pairs are better)
    num_horizontal_pairs = np.sum((cut_left == shift_left) * cut_left)
    num_vertical_pairs = np.sum((cut_down == shift_down) * cut_down)
    num_pairs_score = num_horizontal_pairs + num_vertical_pairs

    # Weight different scores for final result.
    score = empty_tiles_score * EMPTY_TILES_WEIGHT + \
        num_pairs_score * NUM_PAIRS_WEIGHT + \
        max_tile * MAX_TILE_WEIGHT
    return score


# Abbreviation
better = better_evaluation_function
