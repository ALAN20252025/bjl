import sqlite3
import os
import logging
from logging.handlers import RotatingFileHandler

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DB_PATH = os.path.join(DATA_DIR, 'baccarat.db')

def ensure_dir_exists(directory_path):
    """Ensures the specified directory exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

ensure_dir_exists(DATA_DIR)
ensure_dir_exists(LOGS_DIR)


# --- Logger Setup ---
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'app.log')

def setup_logger():
    logger = logging.getLogger('BaccaratApp')
    logger.setLevel(logging.DEBUG) # Set to DEBUG to capture all levels of messages

    # Prevent multiple handlers if setup_logger is called more than once (e.g. in tests)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler - Rotates logs, keeping 5 backups of 5MB each
    fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG) # Log DEBUG and above to file

    # Console Handler - Logs INFO and above to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # Log INFO and above to console

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

app_logger = setup_logger() # Global logger instance for the application


# --- Database Setup ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # ensure_dir_exists(DATA_DIR) # Already called at module level
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Access columns by name
    app_logger.debug(f"Database connection established to {DB_PATH}")
    return conn

def create_tables():
    """Creates the necessary tables in the database if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    app_logger.info("Checking and creating database tables if they don't exist...")

    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS shoes (
            shoe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            number_of_decks INTEGER,
            source TEXT,
            notes TEXT
        );
        """)
        app_logger.debug("Table 'shoes' checked/created.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            round_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shoe_id INTEGER,
            round_number_in_shoe INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            player_cards TEXT,
            banker_cards TEXT,
            player_score INTEGER,
            banker_score INTEGER,
            result TEXT CHECK(result IN ('Player', 'Banker', 'Tie')),
            is_natural INTEGER DEFAULT 0,
            commission_paid REAL,
            raw_input_data TEXT,
            FOREIGN KEY (shoe_id) REFERENCES shoes (shoe_id)
        );
        """)
        app_logger.debug("Table 'rounds' checked/created.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER,
            strategy_name TEXT,
            bet_on TEXT CHECK(bet_on IN ('Player', 'Banker', 'Tie')),
            bet_amount REAL,
            payout REAL,
            is_simulated INTEGER DEFAULT 1,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (round_id) REFERENCES rounds (round_id)
        );
        """)
        app_logger.debug("Table 'bets' checked/created.")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rounds_shoe_id ON rounds (shoe_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_round_id ON bets (round_id);")
        app_logger.debug("Indexes checked/created.")

        conn.commit()
        app_logger.info(f"Database tables checked/created successfully at {DB_PATH}")
    except sqlite3.Error as e:
        app_logger.error(f"Database error during table creation: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_todays_performance_summary():
    """Calculates today's P/L from the bets table."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    app_logger.debug(f"Fetching today's performance summary since {today_start.strftime('%Y-%m-%d %H:%M:%S')}")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Fetch bets made today (based on their timestamp)
        # Ensure your 'bets' table 'timestamp' column is DATETIME and stores in a comparable format.
        # SQLite typically stores DATETIME as TEXT in ISO8601 format or as UNIX timestamp.
        # Assuming ISO8601 format 'YYYY-MM-DD HH:MM:SS'
        cursor.execute("""
            SELECT SUM(payout) as total_payout, COUNT(bet_id) as total_bets
            FROM bets
            WHERE timestamp >= ? AND is_simulated = 1
        """, (today_start.strftime("%Y-%m-%d %H:%M:%S"),)) # Filter for simulated bets today

        row = cursor.fetchone()
        total_payout = row['total_payout'] if row and row['total_payout'] is not None else 0.0
        total_bets = row['total_bets'] if row and row['total_bets'] is not None else 0

        # Note: 'payout' in the bets table is the net win/loss for that bet.
        # So, SUM(payout) is the total P/L.
        app_logger.info(f"Today's performance: Total P/L = {total_payout:.2f}, Total Bets = {total_bets}")
        return total_payout, total_bets

    except sqlite3.Error as e:
        app_logger.error(f"Database error fetching today's performance: {e}", exc_info=True)
        return 0.0, 0 # Return neutral values on error
    except Exception as e:
        app_logger.error(f"Unexpected error fetching today's performance: {e}", exc_info=True)
        return 0.0, 0
    finally:
        conn.close()

if __name__ == '__main__':
    # This will create the db and tables if run directly and test logger
    # ensure_dir_exists(DATA_DIR) # Called at module level
    # ensure_dir_exists(LOGS_DIR) # Called at module level

    app_logger.info("Running utils.py directly for setup and logger test.")
    create_tables()
    app_logger.info("Database initialization complete from utils.py direct run.")
    app_logger.debug("This is a debug message from utils.py direct run.")
    app_logger.warning("This is a warning message from utils.py direct run.")
    app_logger.error("This is an error message from utils.py direct run.")
    # Example usage:
    # conn = get_db_connection()
    # # Do stuff with conn
    # conn.close()

# --- Backup Functionality ---
import zipfile
from datetime import datetime
import time # For caching
import functools # For cache decorator

BACKUP_DIR = os.path.join(BASE_DIR, 'backups')

def backup_data_and_logs():
    """Creates a ZIP backup of the data (baccarat.db) and logs directories."""
    ensure_dir_exists(BACKUP_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = os.path.join(BACKUP_DIR, f"backup_{timestamp}.zip")

    app_logger.info(f"Starting backup process to {backup_filename}...")

    try:
        with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Backup database file
            if os.path.exists(DB_PATH):
                zf.write(DB_PATH, os.path.join('data', os.path.basename(DB_PATH)))
                app_logger.info(f"Backed up database file: {DB_PATH}")
            else:
                app_logger.warning(f"Database file not found at {DB_PATH}, not included in backup.")

            # Backup log files
            if os.path.exists(LOGS_DIR) and os.listdir(LOGS_DIR):
                for root, _, files in os.walk(LOGS_DIR):
                    for file in files:
                        file_path = os.path.join(root, file)
                        archive_name = os.path.relpath(file_path, BASE_DIR) # Store with relative path like 'logs/app.log'
                        zf.write(file_path, archive_name)
                app_logger.info(f"Backed up log files from: {LOGS_DIR}")
            else:
                app_logger.info("Logs directory is empty or does not exist. No logs backed up.")

        app_logger.info(f"Backup completed successfully: {backup_filename}")
        return True
    except Exception as e:
        app_logger.error(f"Error during backup process: {e}", exc_info=True)
        # Clean up partially created backup file if error occurs
        if os.path.exists(backup_filename):
            try:
                os.remove(backup_filename)
                app_logger.info(f"Removed partial backup file: {backup_filename}")
            except OSError as oe:
                app_logger.error(f"Error removing partial backup file {backup_filename}: {oe}")
        return False

if __name__ == '__main__':
    app_logger.info("Running utils.py directly for setup and logger test.")
    create_tables()
    app_logger.info("Database initialization complete from utils.py direct run.")

    app_logger.info("--- Testing Backup Functionality ---")
    if backup_data_and_logs():
        app_logger.info("Backup test successful.")
    else:
        app_logger.error("Backup test failed.")

    app_logger.debug("This is a debug message from utils.py direct run.")
    app_logger.warning("This is a warning message from utils.py direct run.")
    app_logger.error("This is an error message from utils.py direct run.")


# --- Simple Timed Cache Decorator ---
def timed_cache(seconds):
    def decorator(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from args and kwargs (simplified for now)
            # A more robust key would handle mutable args/kwargs or object instances better.
            key_args = tuple(sorted(args))
            key_kwargs = tuple(sorted(kwargs.items()))
            key = (key_args, key_kwargs)

            current_time = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < seconds:
                    app_logger.debug(f"Cache hit for {func.__name__} with key {key}")
                    return result
                else:
                    app_logger.debug(f"Cache expired for {func.__name__} with key {key}")

            app_logger.debug(f"Cache miss for {func.__name__} with key {key}. Calling function.")
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator

# Example of applying it to a function that could be cached
@timed_cache(seconds=60) # Cache for 60 seconds
def get_available_shoe_ids_from_db():
    """Fetches distinct shoe_ids that have rounds, for UI population."""
    app_logger.debug("Executing get_available_shoe_ids_from_db (could be from cache or fresh DB query).")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT s.shoe_id
            FROM shoes s JOIN rounds r ON s.shoe_id = r.shoe_id
            ORDER BY s.shoe_id DESC
        """)
        shoes = [row['shoe_id'] for row in cursor.fetchall()]
        return shoes
    except sqlite3.Error as e:
        app_logger.error(f"Database error fetching available shoe IDs: {e}", exc_info=True)
        return []
    finally:
        conn.close()
