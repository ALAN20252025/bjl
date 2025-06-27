import sys
import os
import logging # For formatter
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import app_logger and create_tables from core.utils
# QtLogHandler and LogDisplayWidget will be imported by MainWindow or here for setup
from core.utils import create_tables, app_logger
from .widgets.log_display_widget import QtLogHandler # Import the handler

# Global reference to the QtLogHandler instance so MainWindow can connect to it
# This is a bit of a shortcut; dependency injection or a dedicated service would be cleaner for larger apps.
global_qt_log_handler = None

def setup_application_logging_for_gui():
    global global_qt_log_handler
    app_logger.info("Application starting up (run_gui.py)...")

    # Create and add the QtLogHandler to the existing app_logger
    global_qt_log_handler = QtLogHandler()
    # Optional: Set a specific format for logs appearing in the GUI
    gui_formatter = logging.Formatter('%(asctime)s [%(levelname)-7s] %(message)s (%(module)s:%(lineno)d)')
    global_qt_log_handler.setFormatter(gui_formatter)
    global_qt_log_handler.setLevel(logging.DEBUG) # Capture all levels for the GUI handler, filtering happens in widget

    app_logger.addHandler(global_qt_log_handler)
    app_logger.info("QtLogHandler added to app_logger. GUI will receive logs.")

    app_logger.info("Ensuring database tables are created...")
    create_tables() # This also logs through app_logger
    app_logger.info("Database tables checked/created.")

    # --- Daily Backup Check ---
    from core.utils import backup_data_and_logs, BACKUP_DIR # Import backup function and dir
    from datetime import date

    today_str = date.today().strftime("%Y%m%d")
    backup_made_today = False
    if os.path.exists(BACKUP_DIR):
        for fname in os.listdir(BACKUP_DIR):
            if fname.startswith(f"backup_{today_str}"):
                backup_made_today = True
                app_logger.info(f"Backup for today ({today_str}) already exists: {fname}")
                break

    if not backup_made_today:
        app_logger.info(f"No backup found for today ({today_str}). Performing backup...")
        backup_data_and_logs()
    # --- End Daily Backup Check ---

try:
    setup_application_logging_for_gui()
except ImportError as e:
    print(f"CRITICAL: Import error during setup: {e}. Application might not function correctly.")
    sys.exit(1)
except Exception as e:
    app_logger.critical(f"Critical error during application startup (logger/DB setup): {e}", exc_info=True)
    sys.exit(1)

# MainWindow must be imported AFTER app_logger and global_qt_log_handler are set up
from .main_window import MainWindow

def main():
    app_logger.info("Initializing QApplication...")
    app = QApplication(sys.argv)

    app_logger.info("Creating MainWindow instance...")
    # Pass the global_qt_log_handler to MainWindow if it needs to connect its LogDisplayWidget
    # Or, MainWindow can import global_qt_log_handler from this module if preferred.
    # For simplicity, let's assume MainWindow will import it or it's passed.
    # Here, we'll rely on MainWindow to instantiate its LogDisplayWidget and connect it.
    # The MainWindow's LogDisplayWidget will need a reference to global_qt_log_handler.new_log_record signal.
    window = MainWindow(qt_log_handler=global_qt_log_handler)

    app_logger.info("Showing MainWindow...")
    window.show()

    app_logger.info("Starting QApplication event loop...")
    exit_code = app.exec()
    app_logger.info(f"Application event loop finished. Exiting with code {exit_code}.")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
