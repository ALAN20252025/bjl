import sqlite3
from datetime import datetime
import time
import random
import threading
from .utils import get_db_connection, app_logger

# --- Existing Functions (start_new_shoe, add_round_to_shoe) ---
def start_new_shoe(number_of_decks=8, source="manual_input", notes=""):
    app_logger.info(f"Attempting to start new shoe. Decks: {number_of_decks}, Source: {source}, Notes: {notes}")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO shoes (number_of_decks, source, notes, start_time)
            VALUES (?, ?, ?, ?)
        """, (number_of_decks, source, notes, datetime.now()))
        conn.commit()
        shoe_id = cursor.lastrowid
        app_logger.info(f"Successfully started new shoe with ID: {shoe_id}")
        return shoe_id
    except sqlite3.Error as e:
        app_logger.error(f"Database error in start_new_shoe: {e}", exc_info=True)
        conn.rollback()
        return None
    finally:
        conn.close()

def add_round_to_shoe(shoe_id, round_number_in_shoe, player_cards, banker_cards, player_score, banker_score, result, is_natural=0, commission_paid=0.0, raw_input_data=""):
    if not shoe_id:
        app_logger.error("Failed to add round: shoe_id is required.")
        return None
    app_logger.info(f"Attempting to add round to shoe {shoe_id}. Round #: {round_number_in_shoe}, P_Score: {player_score}, B_Score: {banker_score}, Result: {result}")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO rounds (shoe_id, round_number_in_shoe, player_cards, banker_cards,
                                player_score, banker_score, result, is_natural,
                                commission_paid, raw_input_data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (shoe_id, round_number_in_shoe, player_cards, banker_cards,
              player_score, banker_score, result, is_natural,
              commission_paid, raw_input_data, datetime.now()))
        conn.commit()
        round_id = cursor.lastrowid
        app_logger.info(f"Successfully added round {round_id} (shoe round {round_number_in_shoe}) to shoe {shoe_id}: Player {player_score} vs Banker {banker_score} -> {result}")
        return round_id
    except sqlite3.Error as e:
        app_logger.error(f"Database error in add_round_to_shoe for shoe_id {shoe_id}: {e}", exc_info=True)
        conn.rollback()
        return None
    finally:
        conn.close()

# --- Real-time Collection Mock ---

MOCK_COLLECTOR_STATE = {
    "is_running": False,
    "active_internal_code": None,
    "current_shoe_id": None,
    "current_round_in_shoe": 0,
    "thread": None,
    "stop_event": threading.Event(),
    "last_round_generated_time": None # For health check
}

def _generate_mock_round_data():
    player_score = random.randint(0, 9)
    banker_score = random.randint(0, 9)
    if player_score > banker_score: result = "Player"
    elif banker_score > player_score: result = "Banker"
    else: result = "Tie"
    player_cards = f"P{random.randint(1,10)},P{random.randint(1,10)}"
    banker_cards = f"B{random.randint(1,10)},B{random.randint(1,10)}"
    is_natural = 1 if player_score >= 8 or banker_score >= 8 else 0
    return {
        "player_cards": player_cards, "banker_cards": banker_cards,
        "player_score": player_score, "banker_score": banker_score,
        "result": result, "is_natural": is_natural
    }

def _mock_real_time_collection_loop():
    app_logger.info(f"Mock real-time collection loop started for internal code: {MOCK_COLLECTOR_STATE['active_internal_code']}, Shoe ID: {MOCK_COLLECTOR_STATE['current_shoe_id']}")
    MOCK_COLLECTOR_STATE["last_round_generated_time"] = time.time() # Initialize on loop start

    while MOCK_COLLECTOR_STATE["is_running"] and not MOCK_COLLECTOR_STATE["stop_event"].is_set():
        try:
            time.sleep(random.uniform(3, 8))
            if MOCK_COLLECTOR_STATE["stop_event"].is_set(): break

            round_data = _generate_mock_round_data()
            MOCK_COLLECTOR_STATE["current_round_in_shoe"] += 1

            app_logger.info(f"[Mock Real-Time] Received new round for Shoe {MOCK_COLLECTOR_STATE['current_shoe_id']}, Round #{MOCK_COLLECTOR_STATE['current_round_in_shoe']}: {round_data['result']}")

            add_round_to_shoe(
                shoe_id=MOCK_COLLECTOR_STATE["current_shoe_id"],
                round_number_in_shoe=MOCK_COLLECTOR_STATE["current_round_in_shoe"],
                player_cards=round_data["player_cards"],
                banker_cards=round_data["banker_cards"],
                player_score=round_data["player_score"],
                banker_score=round_data["banker_score"],
                result=round_data["result"],
                is_natural=round_data["is_natural"],
                commission_paid=0.0,
                raw_input_data=f"Mock real-time data for code {MOCK_COLLECTOR_STATE['active_internal_code']}"
            )
            MOCK_COLLECTOR_STATE["last_round_generated_time"] = time.time() # Update after successful round

            # --- Placeholder for actual web interaction ---
            # ...
            # --- End Placeholder ---

        except Exception as e:
            app_logger.error(f"Error in mock real-time collection loop: {e}", exc_info=True)
            time.sleep(5)

    app_logger.info(f"Mock real-time collection loop stopped for internal code: {MOCK_COLLECTOR_STATE['active_internal_code']}.")


def start_internal_code_session(internal_code: str, number_of_decks=8):
    if MOCK_COLLECTOR_STATE["is_running"]:
        app_logger.warning(f"Cannot start new internal code session. A session with code '{MOCK_COLLECTOR_STATE['active_internal_code']}' is already running.")
        return False

    app_logger.info(f"Attempting to start real-time collection session with internal code: {internal_code}")

    shoe_source = f"realtime_mock_{internal_code}"
    shoe_notes = f"Mock real-time data collection for internal code: {internal_code}"
    current_shoe_id = start_new_shoe(number_of_decks=number_of_decks, source=shoe_source, notes=shoe_notes)

    if not current_shoe_id:
        app_logger.error(f"Failed to start a new shoe for internal code session: {internal_code}. Collector not starting.")
        return False

    MOCK_COLLECTOR_STATE["is_running"] = True
    MOCK_COLLECTOR_STATE["active_internal_code"] = internal_code
    MOCK_COLLECTOR_STATE["current_shoe_id"] = current_shoe_id
    MOCK_COLLECTOR_STATE["current_round_in_shoe"] = 0
    MOCK_COLLECTOR_STATE["stop_event"].clear()
    MOCK_COLLECTOR_STATE["last_round_generated_time"] = time.time() # Initialize timestamp

    # Start the mock collection loop in a separate thread
    # NOTE (Performance): For more complex GUI interactions or frequent updates pushed
    # from the collector to the GUI, consider using QThread and signals/slots
    # for safer cross-thread communication with Qt objects.
    # Current approach (Python threading + QTimer polling in GUI) is okay for this mock.
    MOCK_COLLECTOR_STATE["thread"] = threading.Thread(target=_mock_real_time_collection_loop, daemon=True)
    MOCK_COLLECTOR_STATE["thread"].start()

    app_logger.info(f"Mock real-time collection session started successfully for internal code: {internal_code}. Shoe ID: {current_shoe_id}. Data will be simulated.")
    return True


def stop_internal_code_session():
    if not MOCK_COLLECTOR_STATE["is_running"]:
        app_logger.info("No real-time collection session is currently running.")
        return False

    app_logger.info(f"Attempting to stop real-time collection session for internal code: {MOCK_COLLECTOR_STATE['active_internal_code']}")
    MOCK_COLLECTOR_STATE["is_running"] = False
    MOCK_COLLECTOR_STATE["stop_event"].set()

    if MOCK_COLLECTOR_STATE["thread"] and MOCK_COLLECTOR_STATE["thread"].is_alive():
        app_logger.debug("Waiting for collection thread to finish...")
        MOCK_COLLECTOR_STATE["thread"].join(timeout=10)
        if MOCK_COLLECTOR_STATE["thread"].is_alive():
            app_logger.warning("Collection thread did not finish in time.")
        else:
            app_logger.info("Collection thread finished.")

    if MOCK_COLLECTOR_STATE["current_shoe_id"]:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE shoes SET end_time = ? WHERE shoe_id = ?", (datetime.now(), MOCK_COLLECTOR_STATE["current_shoe_id"]))
            conn.commit()
            conn.close()
            app_logger.info(f"Marked end_time for Shoe ID: {MOCK_COLLECTOR_STATE['current_shoe_id']}")
        except Exception as e:
            app_logger.error(f"Error updating end_time for shoe {MOCK_COLLECTOR_STATE['current_shoe_id']}: {e}", exc_info=True)

    stopped_code = MOCK_COLLECTOR_STATE["active_internal_code"]
    MOCK_COLLECTOR_STATE["active_internal_code"] = None
    MOCK_COLLECTOR_STATE["current_shoe_id"] = None
    MOCK_COLLECTOR_STATE["current_round_in_shoe"] = 0
    MOCK_COLLECTOR_STATE["thread"] = None
    MOCK_COLLECTOR_STATE["last_round_generated_time"] = None

    app_logger.info(f"Real-time collection session stopped for internal code: {stopped_code}")
    return True

def get_collector_status():
    return {
        "is_running": MOCK_COLLECTOR_STATE["is_running"],
        "active_internal_code": MOCK_COLLECTOR_STATE["active_internal_code"],
        "current_shoe_id": MOCK_COLLECTOR_STATE["current_shoe_id"],
        "rounds_collected_in_shoe": MOCK_COLLECTOR_STATE["current_round_in_shoe"],
        "last_round_time": MOCK_COLLECTOR_STATE["last_round_generated_time"]
    }

def check_collector_health(max_idle_seconds=30):
    """Checks if the mock collector seems stuck."""
    status = get_collector_status()
    if status["is_running"]:
        if status["last_round_time"]:
            idle_time = time.time() - status["last_round_time"]
            if idle_time > max_idle_seconds:
                app_logger.warning(f"Collector Health Check: Mock collector has been idle for {idle_time:.2f} seconds (threshold: {max_idle_seconds}s). Potential issue.")
                return False
            else:
                app_logger.debug(f"Collector Health Check: Mock collector is running and recently active (idle for {idle_time:.2f}s).")
                return True
        else:
            app_logger.warning("Collector Health Check: Mock collector is running but last_round_time is not set.")
            return False
    else:
        app_logger.debug("Collector Health Check: Mock collector is not running. Considered healthy/idle.")
        return True

def record_round_from_manual_input(shoe_id):
    app_logger.info(f"Starting manual input for new round for Shoe ID: {shoe_id}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(round_number_in_shoe) FROM rounds WHERE shoe_id = ?", (shoe_id,))
    max_round = cursor.fetchone()[0]
    conn.close()
    next_round_number = (max_round or 0) + 1
    app_logger.debug(f"Next round number in shoe {shoe_id} will be: {next_round_number}")
    print(f"\n--- Adding new round for Shoe ID: {shoe_id}, Round number: {next_round_number} ---")
    try:
        player_cards_str = input("Enter Player cards (e.g., H4,S8): ")
        banker_cards_str = input("Enter Banker cards (e.g., D5,C9,HJ): ")
        player_score = int(input("Enter Player score (0-9): "))
        banker_score = int(input("Enter Banker score (0-9): "))
        valid_results = ['Player', 'Banker', 'Tie']
        result = ""
        while result not in valid_results:
            result = input(f"Enter result ({'/'.join(valid_results)}): ").capitalize()
            if result not in valid_results: print(f"Invalid result. Please enter one of: {', '.join(valid_results)}")
        is_natural_input = input("Was it a natural? (yes/no, default no): ").lower()
        is_natural = 1 if is_natural_input == 'yes' else 0
        commission_paid = 0.0
        app_logger.debug(f"Manual input for shoe {shoe_id}, round {next_round_number}: P_cards={player_cards_str}, B_cards={banker_cards_str}, P_score={player_score}, B_score={banker_score}, Result={result}, Natural={is_natural}")
        return add_round_to_shoe(shoe_id, next_round_number, player_cards_str, banker_cards_str, player_score, banker_score, result, is_natural, commission_paid, f"Manual input: P({player_cards_str}), B({banker_cards_str})")
    except ValueError as ve:
        app_logger.error(f"Invalid input for shoe {shoe_id}: {ve}", exc_info=True)
        print("Invalid input.")
        return None
    except Exception as e:
        app_logger.error(f"Error during manual input for shoe {shoe_id}: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    app_logger.info("Collector Module being run directly for testing mock real-time collection.")
    from .utils import create_tables
    create_tables()

    print("--- Mock Real-Time Collector Test ---")
    test_code = input("Enter a mock internal code to start collection (e.g., 'SITE_A_VIP123'): ")
    if start_internal_code_session(test_code):
        print(f"Mock real-time collection started for code '{test_code}'. Check logs and database.")
        print("Collector will simulate receiving new rounds every few seconds.")
        print("Health checks will run periodically in a real app; here, call manually or integrate into GUI.")
        print("Press Ctrl+C to stop the test (and the collector).")
        try:
            loop_count = 0
            while True:
                status = get_collector_status()
                print(f"Status: Running={status['is_running']}, Code={status['active_internal_code']}, Shoe={status['current_shoe_id']}, Rounds={status['rounds_collected_in_shoe']}, LastActivity={datetime.fromtimestamp(status['last_round_time'] if status['last_round_time'] else time.time()).strftime('%H:%M:%S')}")
                time.sleep(5)
                loop_count+=1
                if loop_count % 6 == 0: # Check health every ~30s
                    check_collector_health()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Stopping mock real-time collection...")
            stop_internal_code_session()
            print("Collection stopped.")
    else:
        print(f"Failed to start mock collection for code '{test_code}'.")

    app_logger.info("Collector direct run finished.")
