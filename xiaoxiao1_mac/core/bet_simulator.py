from .utils import get_db_connection, app_logger
from .strategy_rules import BaseStrategy
from datetime import datetime
import sqlite3
import math # For Kelly Criterion

class BettingSimulator:
    def __init__(self, strategy: BaseStrategy, initial_bankroll=1000,
                 bet_sizing_method="fixed", # "fixed", "percentage", "kelly"
                 fixed_bet_amount=10,
                 percentage_bet_value=0.01, # e.g., 0.01 for 1% of bankroll
                 kelly_fraction=0.1, # e.g., 0.1 for 10% of Kelly fraction
                 stop_loss_percentage=None,
                 stop_profit_percentage=None,
                 max_consecutive_losses=None):

        self.strategy = strategy
        self.initial_bankroll = float(initial_bankroll)
        self.bankroll = float(initial_bankroll)

        # Bet Sizing
        self.bet_sizing_method = bet_sizing_method
        self.fixed_bet_amount = float(fixed_bet_amount)
        self.percentage_bet_value = float(percentage_bet_value)
        self.kelly_fraction = float(kelly_fraction)

        self.bet_history = []
        self.logger = app_logger
        self.consecutive_losses = 0
        self.max_consecutive_losses = max_consecutive_losses

        self.stop_loss_threshold = None
        if stop_loss_percentage is not None and 0 < stop_loss_percentage <= 1:
            self.stop_loss_threshold = self.initial_bankroll * (1 - stop_loss_percentage)

        self.stop_profit_threshold = None
        if stop_profit_percentage is not None and stop_profit_percentage > 0:
            self.stop_profit_threshold = self.initial_bankroll * (1 + stop_profit_percentage)

        self.logger.info(f"Simulator Init: Initial Bankroll: {self.initial_bankroll:.2f}, Bet Sizing: {self.bet_sizing_method}, "
                         f"FixedBet: {self.fixed_bet_amount if self.bet_sizing_method == 'fixed' else 'N/A'}, "
                         f"PercBet: {self.percentage_bet_value*100 if self.bet_sizing_method == 'percentage' else 'N/A'}%, "
                         f"KellyFrac: {self.kelly_fraction if self.bet_sizing_method == 'kelly' else 'N/A'}, "
                         f"SL%: {stop_loss_percentage*100 if stop_loss_percentage else 'N/A'}, "
                         f"SP%: {stop_profit_percentage*100 if stop_profit_percentage else 'N/A'}, "
                         f"MaxConsLoss: {self.max_consecutive_losses if self.max_consecutive_losses else 'N/A'}")

    # get_rounds_for_shoe remains the same
    def get_rounds_for_shoe(self, shoe_id):
        self.logger.debug(f"Simulator: Fetching rounds for shoe_id {shoe_id}.")
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT round_id, shoe_id, round_number_in_shoe, result FROM rounds WHERE shoe_id = ? ORDER BY round_number_in_shoe ASC", (shoe_id,))
            rounds = cursor.fetchall()
            self.logger.debug(f"Simulator: Fetched {len(rounds)} rounds for shoe_id {shoe_id}.")
            return rounds
        except sqlite3.Error as e:
            self.logger.error(f"Simulator: Database error fetching rounds for shoe_id {shoe_id}: {e}", exc_info=True)
            return []
        finally: conn.close()

    # _calculate_payout remains the same
    def _calculate_payout(self, bet_on, actual_result, bet_amount):
        if bet_on == actual_result:
            if bet_on == 'Player': return bet_amount
            elif bet_on == 'Banker': return bet_amount * 0.95 # Standard commission
            elif bet_on == 'Tie': return bet_amount * 8 # Common Tie payout
        return -bet_amount

    # _record_bet_in_db remains the same
    def _record_bet_in_db(self, round_id, strategy_name, bet_on, bet_amount, payout):
        self.logger.debug(f"Simulator: DB Record: Round {round_id}, Strat {strategy_name}, BetOn {bet_on}, Amt {bet_amount:.2f}, Payout {payout:.2f}")
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO bets (round_id, strategy_name, bet_on, bet_amount, payout, is_simulated, timestamp) VALUES (?, ?, ?, ?, ?, 1, ?)",
                           (round_id, strategy_name, bet_on, bet_amount, payout, datetime.now()))
            conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Simulator: DB error recording bet for round {round_id}: {e}", exc_info=True)
            conn.rollback()
        finally: conn.close()

    def _get_current_bet_amount(self, decision_info):
        """Calculates bet amount based on chosen method."""
        if self.bet_sizing_method == "fixed":
            return self.fixed_bet_amount
        elif self.bet_sizing_method == "percentage":
            return max(1.0, self.bankroll * self.percentage_bet_value) # Bet at least 1 unit
        elif self.bet_sizing_method == "kelly":
            # Simplified Kelly: K = (p*b - q) / b
            # p = probability of winning (from strategy if available, else default e.g., ~0.49 for P/B)
            # b = net odds (e.g., 1 for Player, 0.95 for Banker)
            # q = probability of losing (1-p)
            # For this PoC, assume strategy provides 'win_probability' and 'odds' if using Kelly.
            # If not, use a default. This is highly simplified.
            win_prob = decision_info.get('win_probability', 0.493) # Approx P(Banker win non-Tie)
            odds = decision_info.get('odds', 0.95) # Net odds for Banker win

            if odds <= 0: return self.fixed_bet_amount # Avoid division by zero or invalid odds

            kelly_value = (win_prob * odds - (1 - win_prob)) / odds
            if kelly_value <= 0: # If edge is not positive, don't bet or bet minimum
                return 1.0 # Bet minimum if no edge

            bet_size = self.bankroll * kelly_value * self.kelly_fraction
            return max(1.0, min(bet_size, self.bankroll * 0.1)) # Bet at least 1, cap at 10% of bankroll

        self.logger.warning(f"Unknown bet sizing method: {self.bet_sizing_method}. Defaulting to fixed amount: {self.fixed_bet_amount}")
        return self.fixed_bet_amount


    def simulate_shoe(self, shoe_id): # Removed bet_amount_fixed, now determined by bet_sizing_method
        self.logger.info(f"Starting simulation for Shoe ID: {shoe_id} with Strategy: {self.strategy.strategy_name}. Initial Bankroll: {self.initial_bankroll:.2f}")
        rounds_data = self.get_rounds_for_shoe(shoe_id)
        if not rounds_data:
            self.logger.warning(f"No rounds found for shoe ID: {shoe_id}. Simulation cannot proceed.")
            return self.bankroll

        self.reset_simulation(initial_bankroll=self.initial_bankroll)

        for game_round in rounds_data:
            round_num = game_round['round_number_in_shoe']

            if self.stop_loss_threshold is not None and self.bankroll <= self.stop_loss_threshold:
                self.logger.info(f"Shoe {shoe_id}, Round {round_num}: STOP-LOSS triggered. Bankroll ({self.bankroll:.2f}) <= Threshold ({self.stop_loss_threshold:.2f}).")
                break
            if self.stop_profit_threshold is not None and self.bankroll >= self.stop_profit_threshold:
                self.logger.info(f"Shoe {shoe_id}, Round {round_num}: STOP-PROFIT triggered. Bankroll ({self.bankroll:.2f}) >= Threshold ({self.stop_profit_threshold:.2f}).")
                break
            if self.max_consecutive_losses is not None and self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.info(f"Shoe {shoe_id}, Round {round_num}: MAX CONSECUTIVE LOSSES ({self.max_consecutive_losses}) reached. Simulation stopped.")
                break

            # NOTE (Performance): Current strategies are stateless or re-fetch history.
            # For higher performance with very long shoes or complex stateful strategies,
            # consider passing more game state incrementally or making strategies stateful
            # to avoid redundant data fetching/processing per round.
            # This could involve a more complex 'game_context' object passed to decide_bet.
            try:
                decision_info = self.strategy.decide_bet(shoe_id, round_num)
                bet_on = decision_info.get('bet_on')
            except Exception as e:
                self.logger.error(f"S{shoe_id} R{round_num}: Strategy '{self.strategy.strategy_name}' raised an error during decide_bet: {e}", exc_info=True)
                # Decide how to handle: skip round, stop simulation, or default bet (e.g., no bet)
                self.logger.warning(f"S{shoe_id} R{round_num}: Skipping bet due to strategy error.")
                bet_on = None # Default to no bet on strategy error

            actual_result = game_round['result']
            round_id = game_round['round_id']

            if bet_on is None:
                self.logger.info(f"S{shoe_id} R{round_num}: Strategy NO BET. Actual: {actual_result}. Bankroll: {self.bankroll:.2f}")
                self.bet_history.append((round_id, None, 0, actual_result, 0, self.bankroll, self.consecutive_losses))
                continue # No change to consecutive losses if no bet

            current_bet_amount = self._get_current_bet_amount(decision_info)
            if self.bankroll < current_bet_amount:
                self.logger.warning(f"S{shoe_id} R{round_num}: Insufficient bankroll ({self.bankroll:.2f}) for bet ({current_bet_amount:.2f}). Stopping.")
                break

            payout = self._calculate_payout(bet_on, actual_result, current_bet_amount)
            self.bankroll += payout

            if payout < 0: # Loss
                self.consecutive_losses += 1
            else: # Win or push (payout >= 0)
                self.consecutive_losses = 0

            self.bet_history.append((round_id, bet_on, current_bet_amount, actual_result, payout, self.bankroll, self.consecutive_losses))
            self._record_bet_in_db(round_id, self.strategy.strategy_name, bet_on, current_bet_amount, payout)
            self.logger.info(f"S{shoe_id} R{round_num}: Bet {bet_on} ({current_bet_amount:.2f}). Actual: {actual_result}. Payout: {payout:.2f}. Bankroll: {self.bankroll:.2f}. Cons.Loss: {self.consecutive_losses}")

        final_p_l = self.bankroll - self.initial_bankroll
        self.logger.info(f"Sim End S{shoe_id}: Strategy {self.strategy.strategy_name}. Final Bankroll: {self.bankroll:.2f}. Total P/L: {final_p_l:.2f}. Rounds bet: {len(self.bet_history)}")
        return self.bankroll

    def reset_simulation(self, initial_bankroll=None):
        self.bankroll = float(initial_bankroll if initial_bankroll is not None else self.initial_bankroll)

        # Recalculate thresholds based on the *current* initial_bankroll for this run
        current_initial_for_run = float(initial_bankroll if initial_bankroll is not None else self.initial_bankroll)

        if hasattr(self, '_original_stop_loss_percentage') and self._original_stop_loss_percentage is not None:
             self.stop_loss_threshold = current_initial_for_run * (1 - self._original_stop_loss_percentage)
        elif self.stop_loss_threshold is not None: # Store original percentage if first time
            self._original_stop_loss_percentage = 1 - (self.stop_loss_threshold / (self.initial_bankroll + 1e-9))
            self.stop_loss_threshold = current_initial_for_run * (1 - self._original_stop_loss_percentage)

        if hasattr(self, '_original_stop_profit_percentage') and self._original_stop_profit_percentage is not None:
             self.stop_profit_threshold = current_initial_for_run * (1 + self._original_stop_profit_percentage)
        elif self.stop_profit_threshold is not None:
            self._original_stop_profit_percentage = (self.stop_profit_threshold / (self.initial_bankroll + 1e-9)) -1
            self.stop_profit_threshold = current_initial_for_run * (1 + self._original_stop_profit_percentage)

        self.bet_history = []
        self.consecutive_losses = 0
        self.logger.info(f"Simulator reset. Bankroll: {self.bankroll:.2f}. SLT: {self.stop_loss_threshold if self.stop_loss_threshold else 'N/A'}, SPT: {self.stop_profit_threshold if self.stop_profit_threshold else 'N/A'}, MaxCL: {self.max_consecutive_losses if self.max_consecutive_losses else 'N/A'}")


if __name__ == '__main__':
    from .strategy_rules import FollowTheShoeStrategy, RandomStrategy
    from .utils import create_tables
    from .collector import start_new_shoe, add_round_to_shoe
    app_logger.info("Betting Simulator testing advanced risk controls.")
    create_tables()
    # ... (Test data creation as before, ensure enough rounds) ...
    test_shoe_id = start_new_shoe(source="sim_adv_risk_test", notes="Test advanced risk")
    if test_shoe_id:
        for i in range(1, 31): add_round_to_shoe(test_shoe_id, i, "", "", 0,0, random.choice(["Player", "Banker", "Tie"]))

    if not test_shoe_id: app_logger.critical("Failed to create test shoe for advanced risk sim."); exit()

    base_bankroll = 1000
    strat = RandomStrategy() # Use Random for less predictable P/L swings

    app_logger.info("\n--- Test: Fixed Bet, 20% SL, 5 Max Consecutive Losses ---")
    sim1 = BettingSimulator(strategy=strat, initial_bankroll=base_bankroll,
                            bet_sizing_method="fixed", fixed_bet_amount=50,
                            stop_loss_percentage=0.20, max_consecutive_losses=5)
    sim1.simulate_shoe(shoe_id=test_shoe_id)

    app_logger.info("\n--- Test: Percentage Bet (2% of bankroll), 10% SP ---")
    sim2 = BettingSimulator(strategy=strat, initial_bankroll=base_bankroll,
                            bet_sizing_method="percentage", percentage_bet_value=0.02,
                            stop_profit_percentage=0.10)
    sim2.simulate_shoe(shoe_id=test_shoe_id)

    app_logger.info("\n--- Test: Kelly Bet (Simplified, 0.05 fraction), No SL/SP, 3 Max Cons Loss ---")
    # Note: Simplified Kelly needs 'win_probability' and 'odds' from strategy.decide_bet()
    # Our BaseStrategy.decide_bet doesn't provide this. We'd need to enhance strategies or use a fixed default.
    # For now, the Kelly implementation in _get_current_bet_amount has defaults.
    sim3 = BettingSimulator(strategy=strat, initial_bankroll=base_bankroll,
                            bet_sizing_method="kelly", kelly_fraction=0.05,
                            max_consecutive_losses=3)
    sim3.simulate_shoe(shoe_id=test_shoe_id)

    app_logger.info("Advanced risk control tests finished.")
    db_file_path = get_db_connection().execute("PRAGMA database_list;").fetchone()['file']
    app_logger.info(f"Check 'bets' table in '{db_file_path}'.")
