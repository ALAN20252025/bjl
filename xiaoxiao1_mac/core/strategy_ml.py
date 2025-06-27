import random
import numpy as np
import joblib # For saving/loading sklearn models
import os
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.metrics import accuracy_score # For evaluating
from .strategy_rules import BaseStrategy
from .utils import app_logger, get_db_connection, DATA_DIR # DATA_DIR for model storage

# Define a directory for storing trained models
MODELS_DIR = os.path.join(DATA_DIR, 'ml_models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    app_logger.info(f"Created directory for ML models: {MODELS_DIR}")


class SklearnStyleStrategy(BaseStrategy):
    def __init__(self, strategy_name="SklearnStyleBase", model_filename=None):
        super().__init__(strategy_name)
        self.model = None
        self.model_filename = model_filename if model_filename else f"{self.strategy_name.lower().replace(' ', '_')}_model.joblib"
        self.model_path = os.path.join(MODELS_DIR, self.model_filename)
        self.feature_names = ['p_count_l3', 'b_count_l3', 't_count_l3', 'last_winner_is_P', 'last_winner_is_B', 'last_winner_is_T'] # Example feature names
        self.classes_ = ['Player', 'Banker', 'Tie'] # Define the order of classes for predict_proba

        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            self.logger.info(f"{self.strategy_name}: No pre-trained model found at {self.model_path}. Model needs to be trained or loaded.")
            # Initialize a default untrained model
            self.model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced') # Basic default
            self.logger.info(f"{self.strategy_name}: Initialized with a default untrained RandomForestClassifier.")


    def _map_result_to_int(self, result_str):
        return self.classes_.index(result_str) if result_str in self.classes_ else -1 # -1 for unknown/error

    def _map_int_to_result(self, result_int):
        try:
            return self.classes_[result_int]
        except IndexError:
            self.logger.warning(f"Invalid integer {result_int} for class mapping.")
            return None


    def _preprocess_data(self, shoe_id, current_round_number_in_shoe, for_training=False):
        """
        More detailed placeholder for preprocessing.
        If for_training is True, it should return features (X) and labels (y).
        If for_training is False, it returns only features (X) for the current state.
        """
        # Fetch last 3 results (Player, Banker, Tie)
        # In a real scenario, current_round_number_in_shoe would be used to get data *up to* that point.
        # For simplicity in this mock, we'll just use the latest N rounds from the shoe.
        history = self.get_last_n_rounds(shoe_id, n=4 if for_training else 3) # Need 1 more for label if training

        if for_training:
            if len(history) < 4: # Need at least 3 for features, 1 for label
                return None, None
            current_features_history = history[:-1] # Last 3 for features
            label_result = history[-1] # The actual outcome after these features
            y = self._map_result_to_int(label_result)
            if y == -1: return None, None # Skip if label is unknown
        else:
            if len(history) < 3: # Need 3 for features for prediction
                 # Return default features or handle as insufficient data
                self.logger.debug(f"{self.strategy_name}: Insufficient history (need 3, got {len(history)}) for features. Returning zeros.")
                return np.zeros((1, len(self.feature_names)))
            current_features_history = history

        features = np.zeros(len(self.feature_names))
        # Feature 1-3: counts of P, B, T in current_features_history (last 3 rounds)
        p_count = current_features_history.count('Player')
        b_count = current_features_history.count('Banker')
        t_count = current_features_history.count('Tie')
        features[0], features[1], features[2] = p_count, b_count, t_count

        # Feature 4-6: one-hot encode the very last result in current_features_history
        if current_features_history:
            last_res = current_features_history[-1]
            if last_res == 'Player': features[3] = 1
            elif last_res == 'Banker': features[4] = 1
            elif last_res == 'Tie': features[5] = 1

        X = features.reshape(1, -1)
        self.logger.debug(f"{self.strategy_name}: Preprocessed. History: {current_features_history}. Features: {X.tolist()}. Label (if training): {self._map_int_to_result(y) if for_training and y is not None else 'N/A'}")

        if for_training:
            return X, np.array([y]) # y needs to be an array
        return X


    def fit(self, X_train, y_train):
        self.logger.info(f"{self.strategy_name}: Training model with {X_train.shape[0]} samples.")
        if not isinstance(self.model, RandomForestClassifier): # Or your chosen model type
             self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Ensure model is instantiated

        try:
            self.model.fit(X_train, y_train)
            self.classes_ = self.model.classes_ # Store class order from trained model if they are not strings
                                               # If they are strings, ensure self.classes_ matches this order.
                                               # For sklearn classifiers, model.classes_ gives the learned classes.
                                               # We need to map our string labels ('Player', 'Banker', 'Tie') to these.
                                               # For now, assuming our integer mapping (0,1,2) matches what model learns.
            self.logger.info(f"{self.strategy_name}: Model training complete. Learned classes: {self.model.classes_}")
            self.save_model(self.model_path) # Save after training
        except Exception as e:
            self.logger.error(f"{self.strategy_name}: Error during model training: {e}", exc_info=True)
        return self

    def predict_proba(self, X):
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            self.logger.warning(f"{self.strategy_name}: Model not trained/loaded or doesn't support predict_proba. Returning uniform probabilities.")
            num_samples = X.shape[0]
            return np.ones((num_samples, len(self.classes_))) / len(self.classes_)

        try:
            probabilities = self.model.predict_proba(X)
            # Ensure probabilities are in the order of self.classes_
            # This usually aligns if y_train in fit() used 0,1,2 mapped from Player,Banker,Tie
            return probabilities
        except Exception as e:
            self.logger.error(f"{self.strategy_name}: Error during predict_proba: {e}", exc_info=True)
            num_samples = X.shape[0]
            return np.ones((num_samples, len(self.classes_))) / len(self.classes_) # Fallback

    def predict(self, X):
        if self.model is None: # Check if model exists (trained or loaded)
            self.logger.warning(f"{self.strategy_name}: Model not trained/loaded. Cannot predict.")
            # Fallback: make a random choice among Player, Banker, Tie
            return [random.choice(self.classes_) for _ in range(X.shape[0])]

        try:
            predicted_indices = self.model.predict(X) # Returns class indices (0, 1, 2)
            predictions = [self._map_int_to_result(idx) for idx in predicted_indices]
            self.logger.debug(f"{self.strategy_name}: Predicted indices: {predicted_indices}, Mapped predictions: {predictions}")
            return predictions
        except Exception as e:
            self.logger.error(f"{self.strategy_name}: Error during predict: {e}", exc_info=True)
            return [random.choice(self.classes_) for _ in range(X.shape[0])] # Fallback


    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker', 'Tie']
        self.logger.info(f"Strategy {self.strategy_name}: Deciding bet for shoe {shoe_id}, round {current_round_number_in_shoe}. Available: {available_bets}")

        if self.model is None or not hasattr(self.model, 'classes_'): # Check if model seems trained/loaded
            self.logger.warning(f"{self.strategy_name}: Model not properly trained/loaded. Defaulting to random choice.")
            chosen_bet = random.choice(available_bets) if available_bets else None
            return {'bet_on': chosen_bet, 'reason': f"{self.strategy_name}: Model not ready, random choice."}

        features = self._preprocess_data(shoe_id, current_round_number_in_shoe, for_training=False)
        if features is None: # Should not happen if preprocess_data has fallback for insufficient history
             self.logger.warning(f"{self.strategy_name}: Could not generate features. Abstaining.")
             return {'bet_on': None, 'reason': f"{self.strategy_name}: Feature generation failed."}

        predicted_outcome_list = self.predict(features)
        if not predicted_outcome_list:
            self.logger.error(f"{self.strategy_name}: Prediction failed.")
            return {'bet_on': None, 'reason': f"{self.strategy_name}: Prediction error."}
        predicted_bet = predicted_outcome_list[0]

        if predicted_bet not in available_bets:
            self.logger.warning(f"{self.strategy_name}: Predicted '{predicted_bet}', not in {available_bets}. Abstaining.")
            return {'bet_on': None, 'reason': f"{self.strategy_name}: Predicted '{predicted_bet}', not allowed."}

        decision = {'bet_on': predicted_bet, 'reason': f"{self.strategy_name}: Model predicted '{predicted_bet}'."}
        self.logger.debug(f"Strategy {self.strategy_name}: Decision -> {decision}")
        return decision

    def load_model(self, model_path=None):
        path_to_load = model_path if model_path else self.model_path
        try:
            self.model = joblib.load(path_to_load)
            # Ensure self.classes_ is consistent if model stores them differently or if they are implicit
            if hasattr(self.model, 'classes_'):
                 # We need to map string labels to integer indices if the model was trained on integers
                 # And ensure our self.classes_ (P, B, T strings) aligns with model's learned integer classes.
                 # This part can be tricky. For now, assume direct compatibility or simple mapping.
                 # A robust way is to store the class mapping with the model.
                 self.logger.info(f"{self.strategy_name}: Model loaded from {path_to_load}. Learned classes by model: {self.model.classes_}")
            else:
                 self.logger.info(f"{self.strategy_name}: Model loaded from {path_to_load}, but no 'classes_' attribute found on model.")
        except FileNotFoundError:
            self.logger.error(f"{self.strategy_name}: Model file not found at {path_to_load}. Model not loaded.")
            self.model = None # Ensure model is None if loading fails
        except Exception as e:
            self.logger.error(f"{self.strategy_name}: Error loading model from {path_to_load}: {e}", exc_info=True)
            self.model = None


    def save_model(self, model_path=None):
        path_to_save = model_path if model_path else self.model_path
        if self.model and hasattr(self.model, 'fit'): # Check if it's a scikit-learn like model
            try:
                joblib.dump(self.model, path_to_save)
                self.logger.info(f"{self.strategy_name}: Model saved to {path_to_save}.")
            except Exception as e:
                self.logger.error(f"{self.strategy_name}: Error saving model to {path_to_save}: {e}", exc_info=True)
        else:
            self.logger.warning(f"{self.strategy_name}: No model to save or model is not savable.")


class SimpleMLStrategy(SklearnStyleStrategy):
    def __init__(self, model_filename="simple_ml_randomforest_model.joblib"):
        super().__init__(strategy_name="SimpleML_RandomForest", model_filename=model_filename)
        if self.model is None or not self._is_model_fitted(): # If no model loaded and default is not fitted
            self.logger.info(f"{self.strategy_name}: No pre-trained model found or loaded. Initializing a new RandomForestClassifier. Call 'train_on_shoe_data' or 'fit' to train.")
            self.model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')

    def _is_model_fitted(self):
        # Crude check for scikit-learn models; more robust checks might be needed
        return hasattr(self.model, "classes_") and self.model.classes_ is not None


    def train_on_shoe_data(self, shoe_id_list):
        """Example function to gather data from specified shoes and train the model."""
        self.logger.info(f"{self.strategy_name}: Attempting to train model on data from shoes: {shoe_id_list}")
        all_X, all_y = [], []

        conn = get_db_connection()
        cursor = conn.cursor()

        for shoe_id in shoe_id_list:
            # Fetch all rounds for the shoe to construct sequences
            cursor.execute("SELECT round_number_in_shoe, result FROM rounds WHERE shoe_id = ? ORDER BY round_number_in_shoe ASC", (shoe_id,))
            rounds_in_shoe = cursor.fetchall()

            if len(rounds_in_shoe) < 4: # Need at least 3 for features, 1 for label
                self.logger.warning(f"Shoe {shoe_id} has less than 4 rounds, skipping for training.")
                continue

            # Create sequences of (last 3 results as features, next result as label)
            shoe_results_only = [r['result'] for r in rounds_in_shoe]
            for i in range(len(shoe_results_only) - 3):
                history_for_features = shoe_results_only[i : i+3]
                label_for_y = shoe_results_only[i+3]

                # Construct features (same logic as in _preprocess_data)
                features = np.zeros(len(self.feature_names))
                features[0] = history_for_features.count('Player')
                features[1] = history_for_features.count('Banker')
                features[2] = history_for_features.count('Tie')
                if history_for_features: # last of the 3
                    last_res_in_hist = history_for_features[-1]
                    if last_res_in_hist == 'Player': features[3] = 1
                    elif last_res_in_hist == 'Banker': features[4] = 1
                    elif last_res_in_hist == 'Tie': features[5] = 1

                y_val = self._map_result_to_int(label_for_y)
                if y_val != -1:
                    all_X.append(features)
                    all_y.append(y_val)
        conn.close()

        if not all_X or not all_y:
            self.logger.warning(f"{self.strategy_name}: No training data could be generated from shoes {shoe_id_list}. Model not trained.")
            return

        X_train = np.array(all_X)
        y_train = np.array(all_y)

        self.logger.info(f"{self.strategy_name}: Generated {X_train.shape[0]} training samples. Feature shape: {X_train.shape}, Label shape: {y_train.shape}")

        # Optional: Split data if you want to evaluate on a hold-out set
        # X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if len(np.unique(y_train)) > 1 else None)
        # self.fit(X_train_split, y_train_split)
        # self.evaluate(X_test_split, y_test_split) # Implement evaluate method

        self.fit(X_train, y_train) # Train on all generated data

    def evaluate(self, X_test, y_test):
        if self.model and hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            self.logger.info(f"{self.strategy_name}: Evaluation Accuracy on test set: {acc:.4f}")
            return acc
        self.logger.warning(f"{self.strategy_name}: Model not available for evaluation.")
        return None


# --- Example Usage ---
if __name__ == '__main__':
    app_logger.info("strategy_ml.py being run directly for more comprehensive testing.")
    from .utils import create_tables
    from .collector import start_new_shoe, add_round_to_shoe
    create_tables()

    # Create dummy shoes and rounds for training
    shoe1_id = start_new_shoe(source="ml_train_shoe1", notes="Training data")
    if shoe1_id:
        results_s1 = ["Player", "Banker", "Player", "Tie", "Banker", "Banker", "Player"]
        for i, res in enumerate(results_s1): add_round_to_shoe(shoe1_id, i+1, "", "", 0,0, res)

    shoe2_id = start_new_shoe(source="ml_train_shoe2", notes="More training data")
    if shoe2_id:
        results_s2 = ["Banker", "Banker", "Tie", "Player", "Player", "Banker", "Tie", "Player"]
        for i, res in enumerate(results_s2): add_round_to_shoe(shoe2_id, i+1, "", "", 0,0, res)

    # Initialize and train the SimpleMLStrategy
    ml_strategy_instance = SimpleMLStrategy()
    if shoe1_id and shoe2_id:
        ml_strategy_instance.train_on_shoe_data([shoe1_id, shoe2_id])
    else:
        app_logger.warning("Could not create training shoes, ML strategy will use default untrained model or load if exists.")

    # Test decision making on a new shoe (or existing one)
    test_shoe_id_for_ml_decision = start_new_shoe(source="ml_decision_test_shoe", notes="Test decision making")
    if test_shoe_id_for_ml_decision:
        add_round_to_shoe(test_shoe_id_for_ml_decision, 1, "", "", 0,0, "Player")
        add_round_to_shoe(test_shoe_id_for_ml_decision, 2, "", "", 0,0, "Banker")
        add_round_to_shoe(test_shoe_id_for_ml_decision, 3, "", "", 0,0, "Player") # History for decision: P, B, P

        next_round_num = 4
        app_logger.info(f"--- Testing ML Strategy decision for shoe {test_shoe_id_for_ml_decision}, next round {next_round_num} ---")
        decision = ml_strategy_instance.decide_bet(test_shoe_id_for_ml_decision, next_round_num)
        app_logger.info(f"Decision from trained/loaded ML model: {decision}")

        # Test loading a new instance (should load the saved model)
        app_logger.info("--- Testing model loading ---")
        loaded_ml_strategy = SimpleMLStrategy(model_filename=ml_strategy_instance.model_filename) # Uses same filename
        if loaded_ml_strategy.model and hasattr(loaded_ml_strategy.model, 'predict'): # Check if model loaded
            decision_loaded = loaded_ml_strategy.decide_bet(test_shoe_id_for_ml_decision, next_round_num)
            app_logger.info(f"Decision from new instance (loaded model): {decision_loaded}")
        else:
            app_logger.warning("Loaded ML strategy does not seem to have a fitted model.")

    app_logger.info("strategy_ml.py direct run finished.")
