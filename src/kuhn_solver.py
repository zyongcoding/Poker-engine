import random
import numpy as np

class CFRNode:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)

    def get_strategy(self, realization_weight):
        pos_regret = np.maximum(self.regret_sum, 0)
        sum_pos_regret = np.sum(pos_regret)
        
        if sum_pos_regret > 0:
            self.strategy = pos_regret / sum_pos_regret
        else:
            self.strategy = np.ones(self.num_actions) / self.num_actions
            
        self.strategy_sum += realization_weight * self.strategy
        return self.strategy

    def get_average_strategy(self):
        sum_strategy_sum = np.sum(self.strategy_sum)
        if sum_strategy_sum > 0:
            return self.strategy_sum / sum_strategy_sum
        else:
            return np.ones(self.num_actions) / self.num_actions


class KuhnCFRSolver:
    def __init__(self):
        self.node_map = {}

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # 1. Check for Terminal Nodes
        # Pass-Pass (Showdown)
        if history == "pp":
            return 1 if cards[player] > cards[opponent] else -1
            
        # Bet-Call or Pass-Bet-Call (Showdown)
        if history in ["bc", "pbc"]:
            return 2 if cards[player] > cards[opponent] else -2
            
        # Bet-Fold or Pass-Bet-Fold (Fold)
        if history in ["bf", "pbf"]:
            # The player who folded (current player's opponent) loses 1 chip
            return 1

        # 2. Information Set Lookup
        info_set = str(cards[player]) + "_" + history

        if info_set not in self.node_map:
            # 0: Pass/Fold, 1: Bet/Call
            self.node_map[info_set] = CFRNode(num_actions=2)
            
        node = self.node_map[info_set]

        # 3. Strategy Calculation
        current_prob = p0 if player == 0 else p1
        strategy = node.get_strategy(current_prob)

        # 4. Determine Next States and Calculate Utilities
        action_utils = np.zeros(2)
        node_util = 0

        for a in range(2):
            # Explicitly map the next state history
            if history == "":
                next_history = "p" if a == 0 else "b"
            elif history == "p":
                next_history = "pp" if a == 0 else "pb"
            elif history == "b":
                next_history = "bf" if a == 0 else "bc"
            elif history == "pb":
                next_history = "pbf" if a == 0 else "pbc"
            else:
                next_history = history

            # Recursive call with flipped perspectives
            if player == 0:
                action_utils[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                action_utils[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])
                
            node_util += strategy[a] * action_utils[a]

        # 5. Update Regrets
        for a in range(2):
            regret = action_utils[a] - node_util
            opponent_prob = p1 if player == 0 else p0
            node.regret_sum[a] += opponent_prob * regret

        return node_util

    def train(self, iterations):
        cards = [1, 2, 3] # Jack=1, Queen=2, King=3
        for _ in range(iterations):
            random.shuffle(cards)
            self.cfr(cards, "", 1.0, 1.0)

    def print_results(self):
        print("\n=== GTO Strategy Profiles ===")
        print("Format: [Pass/Fold %, Bet/Call %]\n")
        
        card_map = {'1': 'Jack ', '2': 'Queen', '3': 'King '}
        
        for info_set in sorted(self.node_map.keys()):
            node = self.node_map[info_set]
            avg_strat = node.get_average_strategy()
            card, hist = info_set.split('_')
            
            hist_desc = f"History: '{hist}'" if hist != "" else "First to act "
            # Pad strings for a clean console layout
            hist_desc = hist_desc.ljust(15)
            
            print(f"Holding {card_map[card]} | {hist_desc} | Action Frequencies: Fold/Check = {avg_strat[0]:.1%}, Bet/Call = {avg_strat[1]:.1%}")

if __name__ == "__main__":
    solver = KuhnCFRSolver()
    print("Training solver over 100,000 game iterations...")
    solver.train(100000)
    solver.print_results()