import random
from .utils import get_db_connection, app_logger

class BaseStrategy:
    def __init__(self, strategy_name="BaseStrategy"):
        self.strategy_name = strategy_name
        self.logger = app_logger # Use the global logger

    def get_last_n_rounds(self, shoe_id, n=10):
        """Fetches the last n rounds for a given shoe_id."""
        self.logger.debug(f"Strategy {self.strategy_name}: Fetching last {n} rounds for shoe_id {shoe_id}.")
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT result FROM rounds
                WHERE shoe_id = ?
                ORDER BY round_number_in_shoe DESC
                LIMIT ?
            """, (shoe_id, n))
            results = [row['result'] for row in cursor.fetchall()][::-1] # Chronological
            self.logger.debug(f"Strategy {self.strategy_name}: Fetched {len(results)} rounds for shoe_id {shoe_id}: {results}")
            return results
        except Exception as e:
            self.logger.error(f"Strategy {self.strategy_name}: Error fetching rounds for shoe_id {shoe_id}: {e}", exc_info=True)
            return []
        finally:
            conn.close()

    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        """
        Makes a betting decision based on the strategy.
        Returns: A dictionary like {'bet_on': 'Player'/'Banker'/'Tie' or None, 'reason': 'description'}
        """
        if available_bets is None:
            available_bets = ['Player', 'Banker']

        self.logger.info(f"Strategy {self.strategy_name}: Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}. Available: {available_bets}")
        decision = {'bet_on': None, 'reason': "BaseStrategy does not make decisions."}
        self.logger.debug(f"Strategy {self.strategy_name}: Decision for shoe {shoe_id}, round {current_round_number_in_shoe} -> {decision}")
        return decision

class FollowTheShoeStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(strategy_name="FollowTheShoe")

    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
        self.logger.info(f"Strategy {self.strategy_name}: Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}. Available: {available_bets}")

        last_rounds = self.get_last_n_rounds(shoe_id, n=1)
        decision = {'bet_on': None, 'reason': ""}

        if not last_rounds:
            if 'Player' in available_bets:
                 decision = {'bet_on': 'Player', 'reason': "First round or no history, defaulting to Player."}
            elif available_bets:
                decision = {'bet_on': available_bets[0], 'reason': f"First round or no history, defaulting to {available_bets[0]}."}
            else:
                decision = {'bet_on': None, 'reason': "First round or no history, and no available basic bets."}
        else:
            last_winner = last_rounds[0]
            if last_winner in available_bets:
                decision = {'bet_on': last_winner, 'reason': f"Following the shoe, last winner was {last_winner}."}
            else:
                if 'Player' in available_bets:
                     decision = {'bet_on': 'Player', 'reason': f"Last winner {last_winner} not in available bets, defaulting to Player."}
                elif available_bets:
                    decision = {'bet_on': available_bets[0], 'reason': f"Last winner {last_winner} not in available bets, defaulting to {available_bets[0]}."}
                else:
                    decision = {'bet_on': None, 'reason': f"Last winner {last_winner} not in available bets, and no fallback available."}

        self.logger.debug(f"Strategy {self.strategy_name}: Decision for shoe {shoe_id}, round {current_round_number_in_shoe} -> {decision}")
        return decision


class AlternateStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(strategy_name="Alternate")

    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
        self.logger.info(f"Strategy {self.strategy_name}: Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}. Available: {available_bets}")

        last_rounds = self.get_last_n_rounds(shoe_id, n=1)
        bet_decision_val = None
        reason_val = ""

        if not last_rounds:
            if 'Player' in available_bets:
                 bet_decision_val = 'Player'
                 reason_val = "First round or no history, defaulting to Player."
            elif available_bets:
                bet_decision_val = available_bets[0]
                reason_val = f"First round or no history, defaulting to {available_bets[0]}."
            else:
                reason_val = "First round or no history, and no available basic bets."
        else:
            last_winner = last_rounds[0]
            if last_winner == 'Player' and 'Banker' in available_bets:
                bet_decision_val = 'Banker'
                reason_val = "Alternating from Player, betting Banker."
            elif last_winner == 'Banker' and 'Player' in available_bets:
                bet_decision_val = 'Player'
                reason_val = "Alternating from Banker, betting Player."
            elif last_winner == 'Tie':
                if 'Player' in available_bets:
                    bet_decision_val = 'Player'
                    reason_val = "Last was Tie, defaulting to Player."
                elif 'Banker' in available_bets:
                    bet_decision_val = 'Banker'
                    reason_val = "Last was Tie, Player not available, defaulting to Banker."
            else:
                if 'Player' in available_bets:
                     bet_decision_val = 'Player'
                     reason_val = f"Cannot alternate from {last_winner} to preferred, defaulting to Player."
                elif available_bets:
                    bet_decision_val = available_bets[0]
                    reason_val = f"Cannot alternate from {last_winner} to preferred, defaulting to {available_bets[0]}."

        decision = {'bet_on': bet_decision_val, 'reason': reason_val if reason_val else "Could not determine an alternation bet."}
        self.logger.debug(f"Strategy {self.strategy_name}: Decision for shoe {shoe_id}, round {current_round_number_in_shoe} -> {decision}")
        return decision


class RandomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(strategy_name="Random")

    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
        self.logger.info(f"Strategy {self.strategy_name}: Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}. Available: {available_bets}")

        decision = {'bet_on': None, 'reason': ""}
        if not available_bets:
            decision['reason'] = "No available bets to choose from."
        else:
            chosen_bet = random.choice(available_bets)
            decision = {'bet_on': chosen_bet, 'reason': f"Randomly selected {chosen_bet}."}

        self.logger.debug(f"Strategy {self.strategy_name}: Decision for shoe {shoe_id}, round {current_round_number_in_shoe} -> {decision}")
        return decision


# Example Usage (for testing this module directly):
if __name__ == '__main__':
    app_logger.info("Strategy_rules.py being run directly for testing.")

    from .utils import create_tables # This will also setup the logger from utils
    from .collector import start_new_shoe, add_round_to_shoe

    create_tables()

    conn_main_test = get_db_connection()
    cursor_main_test = conn_main_test.cursor()
    cursor_main_test.execute("SELECT shoe_id FROM shoes ORDER BY shoe_id DESC LIMIT 1")
    last_shoe = cursor_main_test.fetchone()
    conn_main_test.close()

    test_shoe_id = None
    if last_shoe:
        test_shoe_id = last_shoe['shoe_id']
        app_logger.info(f"Using existing shoe_id for testing strategies: {test_shoe_id}")
    else:
        app_logger.info("No existing shoes found. Creating a new shoe for testing strategies.")
        test_shoe_id = start_new_shoe(source="strategy_test_direct", notes="Shoe for testing strategies from strategy_rules.py")
        if test_shoe_id:
            add_round_to_shoe(test_shoe_id, 1, "H4,S8", "D5,C9", 2, 4, "Banker")
            add_round_to_shoe(test_shoe_id, 2, "DA,SK", "H7,C2", 1, 9, "Banker")
            add_round_to_shoe(test_shoe_id, 3, "H8,S1", "D8,C1", 9, 9, "Tie")
            add_round_to_shoe(test_shoe_id, 4, "H8,SJ", "D7,CK", 8, 7, "Player")
            add_round_to_shoe(test_shoe_id, 5, "C3,D2", "S5,H6", 5, 1, "Player")
            app_logger.info(f"Created and populated test shoe_id: {test_shoe_id}")
        else:
            app_logger.error("Failed to create a test shoe. Strategy tests might not work as expected.")


    if test_shoe_id:
        app_logger.info(f"--- Testing strategies with Shoe ID: {test_shoe_id} ---")

        follow_strat = FollowTheShoeStrategy()
        alternate_strat = AlternateStrategy()
        random_strat = RandomStrategy()

        last_5_rounds = follow_strat.get_last_n_rounds(test_shoe_id, n=5) # get_last_n_rounds already logs
        app_logger.info(f"Last 5 rounds for shoe {test_shoe_id} (for context): {last_5_rounds}")

        # Simulating decisions for the next round
        next_round_num = len(follow_strat.get_last_n_rounds(test_shoe_id, n=1000)) + 1

        decision_follow = follow_strat.decide_bet(test_shoe_id, next_round_num) # decide_bet logs its own decision
        app_logger.info(f"FollowTheShoe decision result: {decision_follow}")

        decision_alternate = alternate_strat.decide_bet(test_shoe_id, next_round_num)
        app_logger.info(f"AlternateStrategy decision result: {decision_alternate}")

        decision_random = random_strat.decide_bet(test_shoe_id, next_round_num)
        app_logger.info(f"RandomStrategy decision result: {decision_random}")

        decision_random_with_tie = random_strat.decide_bet(test_shoe_id, next_round_num, available_bets=['Player', 'Banker', 'Tie'])
        app_logger.info(f"RandomStrategy (with Tie) decision result: {decision_random_with_tie}")

        new_shoe_id_for_test = start_new_shoe(source="strategy_test_empty_direct", notes="Empty shoe for testing initial strategy decisions")
        if new_shoe_id_for_test:
            app_logger.info(f"--- Testing strategies with NEW Shoe ID: {new_shoe_id_for_test} (no history) ---")
            next_round_num_new_shoe = 1
            decision_follow_new = follow_strat.decide_bet(new_shoe_id_for_test, next_round_num_new_shoe)
            app_logger.info(f"FollowTheShoe (new shoe) decision result: {decision_follow_new}")
            decision_alternate_new = alternate_strat.decide_bet(new_shoe_id_for_test, next_round_num_new_shoe)
            app_logger.info(f"AlternateStrategy (new shoe) decision result: {decision_alternate_new}")
        else:
            app_logger.error("Failed to create a new empty shoe for testing initial strategy decisions.")

    else:
        app_logger.error("Cannot run strategy tests as no shoe_id is available or could be created.")

    app_logger.info("Strategy_rules.py direct run finished.")
