import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QComboBox, QLabel, QHBoxLayout
from PySide6.QtCore import Signal, Slot, QObject

# Custom Handler
class QtLogHandler(logging.Handler, QObject):
    # Signal to emit log messages: new_log_record(level_name, message_string)
    new_log_record = Signal(str, str)

    def __init__(self, parent=None):
        logging.Handler.__init__(self)
        QObject.__init__(self, parent) # QObject part for signals/slots
        # Set a basic formatter, can be overridden by logger's formatter if needed
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.new_log_record.emit(record.levelname, msg)
        except Exception:
            self.handleError(record)


class LogDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_log_level = logging.INFO # Default display level

        self.layout = QVBoxLayout(self)

        # Controls for log level
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Display Log Level:"))
        self.level_combo = QComboBox()
        self.log_levels = {
            "DEBUG": logging.DEBUG, "INFO": logging.INFO,
            "WARNING": logging.WARNING, "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.level_combo.addItems(self.log_levels.keys())
        self.level_combo.setCurrentText("INFO") # Default display level
        self.level_combo.currentTextChanged.connect(self.set_display_level)
        controls_layout.addWidget(self.level_combo)
        controls_layout.addStretch()
        self.layout.addLayout(controls_layout)

        # Log display area
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Optional: prevent line wrapping
        self.layout.addWidget(self.log_text_edit)

        self.log_records_buffer = [] # Store all logs regardless of current display level

    @Slot(str, str)
    def append_log_message(self, level_name, message):
        # Store all messages
        log_entry = (self.log_levels.get(level_name, logging.NOTSET), message)
        self.log_records_buffer.append(log_entry)

        # Display if it meets current filter criteria
        if log_entry[0] >= self.current_log_level:
            self.log_text_edit.append(message)
            # Limit the number of lines to prevent performance issues (optional)
            # max_lines = 1000
            # if self.log_text_edit.document().blockCount() > max_lines:
            #     cursor = self.log_text_edit.textCursor()
            #     cursor.movePosition(QTextCursor.Start)
            #     cursor.select(QTextCursor.LineUnderCursor)
            #     cursor.removeSelectedText()
            #     cursor.deletePreviousChar() # remove the new line character
            # self.log_text_edit.ensureCursorVisible() # Scroll to the bottom

    def set_display_level(self, level_name):
        new_level = self.log_levels.get(level_name, logging.INFO)
        if new_level != self.current_log_level:
            self.current_log_level = new_level
            # Re-filter and display logs from buffer
            self.log_text_edit.clear()
            for level, message in self.log_records_buffer:
                if level >= self.current_log_level:
                    self.log_text_edit.append(message)
            # self.log_text_edit.ensureCursorVisible()

    def clear_logs(self):
        self.log_text_edit.clear()
        self.log_records_buffer.clear()


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys
    import time
    # Fallback basic logger setup for direct testing
    # In the main app, core.utils.app_logger will be the one configured
    test_logger = logging.getLogger("TestLogDisplay")
    test_logger.setLevel(logging.DEBUG)

    # Console handler for seeing logs during test
    ch_test = logging.StreamHandler(sys.stdout)
    ch_test.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    test_logger.addHandler(ch_test)

    app = QApplication(sys.argv)

    log_widget = LogDisplayWidget()

    # Setup custom QtLogHandler and add it to the test_logger
    qt_log_handler = QtLogHandler()
    # Set a specific format for messages going to the GUI log widget
    gui_log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s (%(module)s:%(funcName)s:%(lineno)d)')
    qt_log_handler.setFormatter(gui_log_formatter)

    test_logger.addHandler(qt_log_handler) # Add handler to the logger used in this test

    # Connect the handler's signal to the widget's slot
    qt_log_handler.new_log_record.connect(log_widget.append_log_message)

    log_widget.setWindowTitle("Log Display Test")
    log_widget.resize(800, 400)
    log_widget.show()

    # Emit some test log messages
    test_logger.debug("This is a debug message for the test.")
    test_logger.info("This is an info message for the test.")
    time.sleep(0.1) # ensure messages get processed by event loop if any delay
    test_logger.warning("This is a warning message for the test.")
    test_logger.error("This is an error message for the test.")
    test_logger.critical("This is a critical message for the test.")

    sys.exit(app.exec())
