import json
import os
import importlib
import random
from collections import Counter
from .utils import app_logger, get_db_connection
from .strategy_rules import BaseStrategy # For type hinting and base class if needed

# Define path to strategy configurations
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'strategy_configs')


class StrategyCombiner(BaseStrategy): # Inherit from BaseStrategy for consistent interface
    def __init__(self, config_filename="default_fusion.json"):
        self.config_filename = config_filename
        self.config_path = os.path.join(CONFIG_DIR, self.config_filename)
        
        # Initialize parent class first to get logger
        super().__init__(strategy_name="TemporaryName")
        
        # Now load config and update name
        self.config = self._load_config()
        self.strategy_name = self.config.get("name", "UnnamedFusionStrategy")
        self.logger.info(f"StrategyCombiner initialized with config: {self.config_filename} - Name: {self.strategy_name}")

        self.strategies = self._initialize_strategies(self.config.get("strategies", []))
        self.fallback_strategy = self._initialize_strategies([self.config.get("fallback_strategy")] if self.config.get("fallback_strategy") else [])
        if self.fallback_strategy: # it's a list from _initialize_strategies
            self.fallback_strategy = self.fallback_strategy[0]
            self.logger.info(f"Fallback strategy '{self.fallback_strategy.strategy_name}' initialized.")
        else:
            self.logger.warning("No fallback strategy configured or failed to initialize.")

        self.fusion_method = self.config.get("fusion_method", "majority_vote")
        self.decision_parameters = self.config.get("decision_parameters", {})
        self.logger.info(f"Fusion method: {self.fusion_method}")

    def reload_config(self):
        self.logger.info(f"Attempting to reload config for {self.strategy_name} from {self.config_filename}")
        try:
            new_config = self._load_config() # This re-reads the file

            # Re-initialize based on new config
            # Name might change if config's name field changed, but instance name won't unless re-created.
            # For simplicity, we assume the instance's strategy_name attribute is updated if needed,
            # or that the instance is re-created by whatever manages it (e.g., MainWindow's strategy_map).
            # Here, we'll just update internal components.

            self.config = new_config # Update the stored config
            self.strategy_name = self.config.get("name", f"UnnamedFusionStrategy_{random.randint(100,999)}") # Update name from new config

            self.strategies = self._initialize_strategies(self.config.get("strategies", []))
            fallback_conf = self.config.get("fallback_strategy")
            if fallback_conf:
                temp_fallback_list = self._initialize_strategies([fallback_conf])
                self.fallback_strategy = temp_fallback_list[0] if temp_fallback_list else None
            else:
                self.fallback_strategy = None

            if self.fallback_strategy:
                self.logger.info(f"Fallback strategy '{self.fallback_strategy.strategy_name}' re-initialized for {self.strategy_name}.")
            else:
                self.logger.warning(f"No fallback strategy configured or failed to re-initialize for {self.strategy_name}.")

            self.fusion_method = self.config.get("fusion_method", "majority_vote")
            self.decision_parameters = self.config.get("decision_parameters", {})
            self.logger.info(f"Config for {self.strategy_name} reloaded. New name: {self.strategy_name}, Fusion: {self.fusion_method}. Strategies count: {len(self.strategies)}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload config for {self.strategy_name}: {e}", exc_info=True)
            return False


    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            # Use app_logger directly if self.logger is not available yet
            logger = getattr(self, 'logger', app_logger)
            logger.info(f"Successfully loaded strategy configuration from {self.config_path}")
            return config_data
        except FileNotFoundError:
            logger = getattr(self, 'logger', app_logger)
            logger.error(f"Configuration file not found: {self.config_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            logger = getattr(self, 'logger', app_logger)
            logger.error(f"Error decoding JSON from configuration file: {self.config_path}", exc_info=True)
            raise
        except Exception as e:
            logger = getattr(self, 'logger', app_logger)
            logger.error(f"An unexpected error occurred while loading config {self.config_path}: {e}", exc_info=True)
            raise

    def _initialize_strategies(self, strategy_configs):
        initialized_strategies = []
        for strat_conf in strategy_configs:
            if not strat_conf: continue # Skip if fallback_strategy is not defined leading to [None]
            try:
                module_name = strat_conf.get("module")
                class_name = strat_conf.get("class")
                # strategy_params = strat_conf.get("params", {}) # For future use if strategies take __init__ params

                if not module_name or not class_name:
                    self.logger.error(f"Strategy config missing 'module' or 'class': {strat_conf}")
                    continue

                module = importlib.import_module(module_name)
                StrategyClass = getattr(module, class_name)
                strategy_instance = StrategyClass() # **strategy_params if params were used

                # Store weight if provided, for weighted voting later
                strategy_instance.weight = strat_conf.get("weight", 1)
                initialized_strategies.append(strategy_instance)
                self.logger.info(f"Successfully initialized strategy: {strategy_instance.strategy_name} from {module_name}.{class_name} with weight {strategy_instance.weight}")
            except ImportError:
                self.logger.error(f"Could not import module {module_name} for strategy {class_name}", exc_info=True)
            except AttributeError:
                self.logger.error(f"Could not find class {class_name} in module {module_name}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error initializing strategy {class_name}: {e}", exc_info=True)
        return initialized_strategies

    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        self.logger.info(f"Combiner '{self.strategy_name}': Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}")

        # Use available_bets from config if not provided, else use argument, else default
        if available_bets is None:
            available_bets = self.decision_parameters.get('available_bets', ['Player', 'Banker'])

        decisions = []
        for strat in self.strategies:
            try:
                decision_info = strat.decide_bet(shoe_id, current_round_number_in_shoe, available_bets)
                if decision_info and decision_info.get('bet_on') is not None:
                    # For weighted voting, repeat the decision by its weight
                    for _ in range(int(strat.weight)):
                        decisions.append(decision_info['bet_on'])
                    self.logger.debug(f"Strategy '{strat.strategy_name}' (weight {strat.weight}) decided: {decision_info.get('bet_on')}. Reason: {decision_info.get('reason')}")
                else:
                    self.logger.debug(f"Strategy '{strat.strategy_name}' (weight {strat.weight}) abstained or no decision.")
            except Exception as e:
                self.logger.error(f"Error getting decision from strategy '{strat.strategy_name}': {e}", exc_info=True)

        if not decisions:
            self.logger.warning("No decisions made by any of the primary strategies.")
            if self.fallback_strategy:
                self.logger.info(f"Using fallback strategy: {self.fallback_strategy.strategy_name}")
                return self.fallback_strategy.decide_bet(shoe_id, current_round_number_in_shoe, available_bets)
            else:
                self.logger.warning("No decisions and no fallback strategy. Abstaining.")
                return {'bet_on': None, 'reason': "All primary strategies abstained, no fallback."}

        # --- Fusion Logic ---
        final_decision_val = None
        reason_val = ""

        if self.fusion_method == "majority_vote" or self.fusion_method == "majority_vote_with_fallback": # Handle fallback outside
            self.logger.debug(f"Applying fusion method: {self.fusion_method}. Decisions received: {decisions}")
            vote_counts = Counter(decisions)
            self.logger.debug(f"Vote counts: {vote_counts.most_common()}")

            if not vote_counts: # Should be caught by 'if not decisions:' above, but as a safeguard
                 self.logger.warning("Majority vote called with no decisions.")
                 # Fallback logic will be handled outside if method is 'majority_vote_with_fallback'
                 final_decision_val = None
                 reason_val = "No votes from primary strategies."

            else:
                # Check for a clear majority or a tie
                most_common_votes = vote_counts.most_common()
                if len(most_common_votes) == 1: # Only one type of vote, clear majority
                    final_decision_val = most_common_votes[0][0]
                    reason_val = f"Unanimous vote for {final_decision_val} (count: {most_common_votes[0][1]})."
                elif most_common_votes[0][1] > most_common_votes[1][1]: # Clear majority
                    final_decision_val = most_common_votes[0][0]
                    reason_val = f"Majority vote for {final_decision_val} (count: {most_common_votes[0][1]}). Votes: {vote_counts}"
                else: # Tie for the highest vote
                    self.logger.info(f"Tie in majority vote: {most_common_votes}. ")
                    if self.fusion_method == "majority_vote_with_fallback" and self.fallback_strategy:
                        self.logger.info(f"Tie detected, using fallback strategy: {self.fallback_strategy.strategy_name}")
                        return self.fallback_strategy.decide_bet(shoe_id, current_round_number_in_shoe, available_bets)
                    else: # No fallback or not using it for ties
                        # Could pick one randomly, or abstain. Let's abstain on tie without specific tie-breaker.
                        final_decision_val = None
                        reason_val = f"Tie in voting ({vote_counts}), no tie-breaking rule or fallback for tie. Abstaining."

        # Add other fusion methods here if needed (e.g., weighted_average_prob if strategies output probabilities)
        else:
            self.logger.error(f"Unsupported fusion_method: {self.fusion_method}. Abstaining.")
            final_decision_val = None
            reason_val = f"Unsupported fusion method: {self.fusion_method}."

        if final_decision_val and final_decision_val not in available_bets:
            self.logger.warning(f"Combined decision '{final_decision_val}' is not in available_bets {available_bets}. Overriding to abstain.")
            reason_val = f"Combined decision '{final_decision_val}' not in {available_bets}. Original reason: {reason_val}"
            final_decision_val = None


        final_decision_info = {'bet_on': final_decision_val, 'reason': reason_val}
        self.logger.info(f"StrategyCombiner '{self.strategy_name}' final decision: {final_decision_info}")
        return final_decision_info


# --- Example Usage ---
if __name__ == '__main__':
    app_logger.info("StrategyCombiner being run directly for testing.")

    from .utils import create_tables # Ensures logger and DB are ready
    from .collector import start_new_shoe, add_round_to_shoe # For test data
    # Ensure ML strategy has a "trained" model for fallback if used by config
    from .strategy_ml import SimpleMLStrategy
    ml_strat_for_combiner_test = SimpleMLStrategy()
    ml_strat_for_combiner_test.fit(None,None) # "Train" the dummy model

    create_tables()

    # Ensure a shoe with some history exists
    conn_test = get_db_connection()
    cursor_test = conn_test.cursor()
    # Check for a shoe with at least 1 round for strategies to have some history
    cursor_test.execute("SELECT shoe_id FROM rounds GROUP BY shoe_id HAVING COUNT(*) >= 1 ORDER BY shoe_id DESC LIMIT 1")
    shoe_row = cursor_test.fetchone()
    conn_test.close()

    test_shoe_id_combiner = None
    if shoe_row:
        test_shoe_id_combiner = shoe_row['shoe_id']
        app_logger.info(f"Using existing shoe_id for combiner test: {test_shoe_id_combiner}")
    else:
        app_logger.info("No suitable shoe for combiner test. Creating one.")
        test_shoe_id_combiner = start_new_shoe(source="combiner_test_direct", notes="Shoe for combiner testing")
        if test_shoe_id_combiner:
            add_round_to_shoe(test_shoe_id_combiner, 1, "", "", 0, 0, "Player") # Last was Player
            # add_round_to_shoe(test_shoe_id_combiner, 2, "", "", 0, 0, "Banker")
            app_logger.info(f"Created and populated shoe {test_shoe_id_combiner} for combiner testing.")
        else:
            app_logger.error("Failed to create shoe for combiner testing.")
            test_shoe_id_combiner = None

    if test_shoe_id_combiner:
        app_logger.info(f"--- Testing StrategyCombiner with config 'default_fusion.json' on shoe {test_shoe_id_combiner} ---")

        # Get current number of rounds to simulate decision for the next one
        conn_temp = get_db_connection()
        cursor_temp = conn_temp.cursor()
        cursor_temp.execute("SELECT COUNT(round_id) FROM rounds WHERE shoe_id = ?", (test_shoe_id_combiner,))
        num_rounds_in_shoe = cursor_temp.fetchone()[0]
        conn_temp.close()
        next_round_for_combiner = num_rounds_in_shoe + 1

        try:
            combiner_strategy = StrategyCombiner(config_filename="default_fusion.json")
            # For default_fusion.json with last round "Player":
            # FollowTheShoe -> Player
            # Alternate -> Banker
            # Fallback (ML) -> Random (as it's not really trained)
            # Expected: Tie, so fallback to ML strategy's random choice.

            combiner_decision = combiner_strategy.decide_bet(test_shoe_id_combiner, next_round_for_combiner)
            app_logger.info(f"Combiner strategy 'default_fusion.json' decision: {combiner_decision}")

            # Test with different available_bets
            combiner_decision_pb_only = combiner_strategy.decide_bet(test_shoe_id_combiner, next_round_for_combiner, available_bets=['Player', 'Banker'])
            app_logger.info(f"Combiner strategy 'default_fusion.json' (Player/Banker only) decision: {combiner_decision_pb_only}")

        except Exception as e:
            app_logger.error(f"Error during StrategyCombiner test: {e}", exc_info=True)

    app_logger.info("StrategyCombiner direct run finished.")
