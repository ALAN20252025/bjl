from PySide6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget,
                               QPushButton, QTableView, QHBoxLayout, QComboBox,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
                               QMessageBox, QCheckBox, QSplitter, QLineEdit, QTabWidget,
                               QTextEdit, QSystemTrayIcon, QMenu) # Added QTextEdit, QSystemTrayIcon, QMenu
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon, QAction
from PySide6.QtCore import Qt, QTimer
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.utils import get_db_connection, app_logger, get_todays_performance_summary # Import new util
from core.strategy_rules import FollowTheShoeStrategy, AlternateStrategy, RandomStrategy
from core.strategy_ml import SimpleMLStrategy, RandomForestClassifier
from core.strategy_combiner import StrategyCombiner
from core.bet_simulator import BettingSimulator
from core.bet_manager import BetManager # Import BetManager
from core import collector
from .widgets.simulation_chart_widget import SimulationChartWidget
from .widgets.log_display_widget import LogDisplayWidget

class MainWindow(QMainWindow):
    def __init__(self, qt_log_handler=None):
        super().__init__()
        self.logger = app_logger
        self.logger.info("MainWindow initialized.")
        self.setWindowTitle("XiaoXiao1 Baccarat Assistant")
        self.setGeometry(100, 100, 1400, 900)

        self.qt_log_handler = qt_log_handler

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel_widget = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_panel_widget)
        self._setup_control_panel_group()
        self._setup_real_bet_panel_group()
        self._setup_simulation_group()
        self._setup_results_group()
        self._setup_auxiliary_area_group()
        self._setup_hot_reload_group() # New group for hot reloading
        self.left_panel_layout.addStretch()
        self.left_panel_widget.setFixedWidth(400)
        self.main_layout.addWidget(self.left_panel_widget)

        self.right_tab_widget = QTabWidget()
        self.data_chart_tab = QWidget()
        self.data_chart_tab_layout = QVBoxLayout(self.data_chart_tab)

        self.right_splitter = QSplitter(Qt.Vertical)
        self.data_display_widget = QWidget()
        self.data_display_layout = QVBoxLayout(self.data_display_widget)
        self._setup_data_display_controls()
        self.rounds_table_view = QTableView()
        self.rounds_table_model = QStandardItemModel()
        self.rounds_table_view.setModel(self.rounds_table_model)
        self.data_display_layout.addWidget(self.rounds_table_view)
        self.right_splitter.addWidget(self.data_display_widget)

        self.chart_widget = SimulationChartWidget()
        self.right_splitter.addWidget(self.chart_widget)
        self.right_splitter.setSizes([400, 300])

        self.data_chart_tab_layout.addWidget(self.right_splitter)
        self.right_tab_widget.addTab(self.data_chart_tab, "📊 Data & Charts")

        self.log_tab = QWidget()
        self.log_tab_layout = QVBoxLayout(self.log_tab)
        self.log_display_widget = LogDisplayWidget()
        if self.qt_log_handler:
            self.qt_log_handler.new_log_record.connect(self.log_display_widget.append_log_message)
            self.logger.info("Connected QtLogHandler to LogDisplayWidget.")
        else:
            self.logger.warning("No QtLogHandler provided to MainWindow. Log display in GUI will not work.")
        self.log_tab_layout.addWidget(self.log_display_widget)
        self.right_tab_widget.addTab(self.log_tab, "🧾 Logs")

        self.main_layout.addWidget(self.right_tab_widget)

        self.load_styles()
        self._update_strategy_map()
        self.load_round_data()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_collector_status_display)
        self.status_timer.timeout.connect(self._update_auxiliary_info) # Also update aux info periodically
        self.status_timer.start(5000) # Check status less frequently, e.g., every 5s
        self.update_collector_status_display()
        self._update_auxiliary_info() # Initial update

        self._setup_tray_icon()

    def _setup_control_panel_group(self):
        control_panel_group = QGroupBox("🎛 Real-time Control Panel")
        cp_layout = QFormLayout()
        self.internal_code_input = QLineEdit()
        self.internal_code_input.setPlaceholderText("Enter Internal Code")
        cp_layout.addRow("Internal Code:", self.internal_code_input)
        self.start_collection_button = QPushButton("Start Real-time Collection")
        self.start_collection_button.clicked.connect(self.start_realtime_collection)
        self.stop_collection_button = QPushButton("Stop Real-time Collection")
        self.stop_collection_button.clicked.connect(self.stop_realtime_collection)
        self.stop_collection_button.setEnabled(False)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_collection_button)
        button_layout.addWidget(self.stop_collection_button)
        cp_layout.addRow(button_layout)
        self.collector_status_label = QLabel("Collector Status: Idle")
        self.collector_status_label.setWordWrap(True)
        cp_layout.addRow(self.collector_status_label)
        control_panel_group.setLayout(cp_layout)
        self.left_panel_layout.addWidget(control_panel_group)

    def _update_strategy_map(self):
        self.strategy_map = {
            "Follow The Shoe": FollowTheShoeStrategy(), "Alternate": AlternateStrategy(),
            "Random Rule-Based": RandomStrategy(), "Simple ML (Placeholder)": SimpleMLStrategy(),
        }
        try:
            default_combiner = StrategyCombiner(config_filename="default_fusion.json")
            if default_combiner.fallback_strategy and isinstance(default_combiner.fallback_strategy, SimpleMLStrategy):
                if not default_combiner.fallback_strategy.model: default_combiner.fallback_strategy.fit(None,None)
            self.strategy_map[default_combiner.strategy_name] = default_combiner
            self.logger.info(f"Loaded combiner strategy: {default_combiner.strategy_name}")
        except FileNotFoundError: self.logger.warning("default_fusion.json not found.")
        except Exception as e: self.logger.error(f"Error loading default_fusion.json: {e}", exc_info=True)

        current_sim_strategy_text = self.strategy_combo.currentText() if hasattr(self, 'strategy_combo') else None
        if hasattr(self, 'strategy_combo'):
            self.strategy_combo.clear()
            self.strategy_combo.addItems(self.strategy_map.keys())
            if current_sim_strategy_text in self.strategy_map:
                self.strategy_combo.setCurrentText(current_sim_strategy_text)

    def _setup_simulation_group(self):
        simulation_group = QGroupBox("📈 Simulation Setup")
        simulation_form_layout = QFormLayout()
        self.shoe_id_combo = QComboBox()
        simulation_form_layout.addRow("Select Shoe ID:", self.shoe_id_combo)
        self.strategy_combo = QComboBox()
        simulation_form_layout.addRow("Select Strategy:", self.strategy_combo)
        self.initial_bankroll_spin = QDoubleSpinBox()
        self.initial_bankroll_spin.setRange(0, 1000000); self.initial_bankroll_spin.setValue(1000); self.initial_bankroll_spin.setSuffix(" $")
        simulation_form_layout.addRow("Initial Bankroll:", self.initial_bankroll_spin)

        self.bet_sizing_combo = QComboBox()
        self.bet_sizing_combo.addItems(["Fixed", "Percentage", "Kelly"])
        self.bet_sizing_combo.currentTextChanged.connect(self._update_bet_sizing_inputs_visibility)
        simulation_form_layout.addRow("Bet Sizing Method:", self.bet_sizing_combo)

        self.fixed_bet_amount_label = QLabel("Fixed Bet Amount:")
        self.bet_amount_spin = QDoubleSpinBox()
        self.bet_amount_spin.setRange(1, 10000); self.bet_amount_spin.setValue(10); self.bet_amount_spin.setSuffix(" $")
        simulation_form_layout.addRow(self.fixed_bet_amount_label, self.bet_amount_spin)

        self.percentage_bet_label = QLabel("Bet Percentage (% of Bankroll):")
        self.percentage_bet_spin = QDoubleSpinBox()
        self.percentage_bet_spin.setRange(0.01, 100.0); self.percentage_bet_spin.setValue(1.0); self.percentage_bet_spin.setSuffix(" %"); self.percentage_bet_spin.setDecimals(2)
        simulation_form_layout.addRow(self.percentage_bet_label, self.percentage_bet_spin)

        self.kelly_fraction_label = QLabel("Kelly Fraction (0.01-1.0):")
        self.kelly_fraction_spin = QDoubleSpinBox()
        self.kelly_fraction_spin.setRange(0.01, 1.0); self.kelly_fraction_spin.setValue(0.1); self.kelly_fraction_spin.setDecimals(2)
        simulation_form_layout.addRow(self.kelly_fraction_label, self.kelly_fraction_spin)

        self._update_bet_sizing_inputs_visibility(self.bet_sizing_combo.currentText())

        self.max_consecutive_losses_label = QLabel("Max Cons. Losses (0 to disable):")
        self.max_consecutive_losses_spin = QSpinBox()
        self.max_consecutive_losses_spin.setRange(0, 100); self.max_consecutive_losses_spin.setValue(0)
        simulation_form_layout.addRow(self.max_consecutive_losses_label, self.max_consecutive_losses_spin)

        self.enable_stop_loss_check = QCheckBox("Enable Stop-Loss")
        self.stop_loss_percentage_spin = QSpinBox()
        self.stop_loss_percentage_spin.setRange(1, 100); self.stop_loss_percentage_spin.setValue(20); self.stop_loss_percentage_spin.setSuffix(" %")
        self.stop_loss_percentage_spin.setEnabled(False)
        self.enable_stop_loss_check.toggled.connect(self.stop_loss_percentage_spin.setEnabled)
        sl_layout = QHBoxLayout(); sl_layout.addWidget(self.enable_stop_loss_check); sl_layout.addWidget(self.stop_loss_percentage_spin)
        simulation_form_layout.addRow(sl_layout)

        self.enable_stop_profit_check = QCheckBox("Enable Stop-Profit")
        self.stop_profit_percentage_spin = QSpinBox()
        self.stop_profit_percentage_spin.setRange(1, 500); self.stop_profit_percentage_spin.setValue(50); self.stop_profit_percentage_spin.setSuffix(" %")
        self.stop_profit_percentage_spin.setEnabled(False)
        self.enable_stop_profit_check.toggled.connect(self.stop_profit_percentage_spin.setEnabled)
        sp_layout = QHBoxLayout(); sp_layout.addWidget(self.enable_stop_profit_check); sp_layout.addWidget(self.stop_profit_percentage_spin)
        simulation_form_layout.addRow(sp_layout)

        self.run_simulation_button = QPushButton("Run Simulation")
        self.run_simulation_button.clicked.connect(self.run_simulation)
        simulation_form_layout.addRow(self.run_simulation_button)
        simulation_group.setLayout(simulation_form_layout)
        self.left_panel_layout.addWidget(simulation_group)

    def _setup_results_group(self):
        results_group = QGroupBox("📊 Simulation Results")
        results_layout = QFormLayout()
        self.result_strategy_label = QLabel("N/A")
        results_layout.addRow("Strategy Used:", self.result_strategy_label)
        self.result_shoe_id_label = QLabel("N/A")
        results_layout.addRow("Simulated Shoe ID:", self.result_shoe_id_label)
        self.result_initial_bankroll_label = QLabel("N/A")
        results_layout.addRow("Initial Bankroll:", self.result_initial_bankroll_label)
        self.result_final_bankroll_label = QLabel("N/A")
        results_layout.addRow("Final Bankroll:", self.result_final_bankroll_label)
        self.result_profit_loss_label = QLabel("N/A")
        results_layout.addRow("Total Profit/Loss:", self.result_profit_loss_label)
        self.result_stop_reason_label = QLabel("N/A")
        results_layout.addRow("Stop Reason:", self.result_stop_reason_label)
        results_group.setLayout(results_layout)
        self.left_panel_layout.addWidget(results_group)

    def _setup_real_bet_panel_group(self):
        real_bet_group = QGroupBox("💸 Simulated Real Bet")
        rb_layout = QFormLayout()

        self.rb_shoe_id_label = QLabel("N/A (Auto from Collector)") # Display current shoe from collector
        rb_layout.addRow("Current Shoe ID:", self.rb_shoe_id_label)

        self.rb_round_number_label = QLabel("N/A (Auto from Collector)")
        rb_layout.addRow("Current Round #:", self.rb_round_number_label)

        self.rb_bet_on_combo = QComboBox()
        self.rb_bet_on_combo.addItems(["Player", "Banker", "Tie"])
        rb_layout.addRow("Bet On:", self.rb_bet_on_combo)

        self.rb_bet_amount_spin = QDoubleSpinBox()
        self.rb_bet_amount_spin.setRange(1, 10000); self.rb_bet_amount_spin.setValue(10); self.rb_bet_amount_spin.setSuffix(" $")
        rb_layout.addRow("Bet Amount:", self.rb_bet_amount_spin)

        # Strategy for this bet (can be manual or from a live-selected strategy)
        self.rb_strategy_label = QLabel("ManualLiveBet") # Default
        # Potentially link this to a live strategy selector later
        # For now, it's fixed or could be an input field.
        rb_layout.addRow("Bet Source:", self.rb_strategy_label)


        self.place_real_bet_button = QPushButton("Place Simulated Real Bet")
        self.place_real_bet_button.clicked.connect(self.place_simulated_real_bet)
        rb_layout.addRow(self.place_real_bet_button)

        self.real_bet_status_label = QLabel("Status: Ready")
        rb_layout.addRow(self.real_bet_status_label)

        real_bet_group.setLayout(rb_layout)
        self.left_panel_layout.addWidget(real_bet_group)

        # Initialize BetManager instance
        self.bet_manager = BetManager()

    def _setup_data_display_controls(self):
        data_controls_layout = QHBoxLayout()
        self.refresh_rounds_button = QPushButton("Refresh Rounds Data")
        self.refresh_rounds_button.clicked.connect(self.load_round_data)
        data_controls_layout.addWidget(self.refresh_rounds_button)
        data_controls_layout.addStretch()
        self.data_display_layout.addLayout(data_controls_layout)

    def load_styles(self):
        self.logger.debug("Loading styles...")
        try:
            style_path = os.path.join(os.path.dirname(__file__), "style.qss")
            with open(style_path, "r") as f: self.setStyleSheet(f.read())
            self.logger.info("Styles loaded successfully.")
        except FileNotFoundError: self.logger.warning(f"Style file not found: {style_path}")
        except Exception as e: self.logger.error(f"Error loading styles: {e}", exc_info=True)

    def load_available_shoes(self):
        current_shoe_text = self.shoe_id_combo.currentText()
        self.logger.debug("Loading available shoes into combobox.")
        self.shoe_id_combo.clear()
        try:
            conn = get_db_connection(); cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT s.shoe_id FROM shoes s JOIN rounds r ON s.shoe_id = r.shoe_id ORDER BY s.shoe_id DESC")
            shoes = cursor.fetchall(); conn.close()
            if shoes:
                for shoe in shoes: self.shoe_id_combo.addItem(str(shoe['shoe_id']))
                self.logger.info(f"Loaded {len(shoes)} shoe IDs into combobox.")
                if current_shoe_text in [str(s['shoe_id']) for s in shoes]: self.shoe_id_combo.setCurrentText(current_shoe_text)
            else:
                self.shoe_id_combo.addItem("No shoes with data")
                self.logger.info("No shoes with data found to load.")
        except Exception as e:
            self.logger.error(f"Error loading available shoes: {e}", exc_info=True)
            self.shoe_id_combo.addItem("Error loading shoes")

    def load_round_data(self):
        self.logger.info("Loading round data into table view.")
        self.rounds_table_model.clear()
        headers = ["Round ID", "Shoe ID", "Round #", "Timestamp", "P Cards", "B Cards", "P Score", "B Score", "Result", "Natural?", "Comm.", "Raw Data"]
        self.rounds_table_model.setHorizontalHeaderLabels(headers)
        try:
            conn = get_db_connection(); cursor = conn.cursor()
            cursor.execute("SELECT round_id, shoe_id, round_number_in_shoe, timestamp, player_cards, banker_cards, player_score, banker_score, result, is_natural, commission_paid, raw_input_data FROM rounds ORDER BY shoe_id DESC, round_number_in_shoe DESC LIMIT 200")
            rounds_data = cursor.fetchall(); conn.close()
            for row_data in rounds_data:
                row_items = [QStandardItem(str(row_data[i] if row_data[i] is not None else "")) for i in range(len(headers))]
                for item in row_items: item.setEditable(False)
                self.rounds_table_model.appendRow(row_items)
            self.rounds_table_view.resizeColumnsToContents()
            self.logger.info(f"Loaded {len(rounds_data)} rounds into table view.")
            self.load_available_shoes()
        except Exception as e:
            self.logger.error(f"Error loading round data: {e}", exc_info=True)
            self.rounds_table_model.appendRow([QStandardItem(f"Error loading data: {e}")])

    def start_realtime_collection(self):
        internal_code = self.internal_code_input.text().strip()
        if not internal_code:
            QMessageBox.warning(self, "Input Error", "Please enter an internal code.")
            self.logger.warning("Start collection attempt failed: No internal code provided.")
            return
        self.logger.info(f"GUI: Attempting to start real-time collection with code: {internal_code}")
        if collector.start_internal_code_session(internal_code):
            self.start_collection_button.setEnabled(False)
            self.stop_collection_button.setEnabled(True)
            self.internal_code_input.setEnabled(False)
            QMessageBox.information(self, "Collection Started", f"Mock real-time data collection started for code: {internal_code}")
            self.logger.info(f"GUI: Mock collection started successfully for code {internal_code}.")
            self._show_tray_message("Collection Started", f"Mock data collection started for code: {internal_code}.")
        else:
            QMessageBox.critical(self, "Collection Error", f"Failed to start collection for code: {internal_code}. See logs for details.")
            self.logger.error(f"GUI: Failed to start collection for code {internal_code}.")
            self._show_tray_message("Collection Error", f"Failed to start collection for {internal_code}.", icon=QSystemTrayIcon.Icon.Critical)
        self.update_collector_status_display()

    def stop_realtime_collection(self):
        self.logger.info("GUI: Attempting to stop real-time collection.")
        if collector.stop_internal_code_session():
            self.start_collection_button.setEnabled(True)
            self.stop_collection_button.setEnabled(False)
            self.internal_code_input.setEnabled(True)
            QMessageBox.information(self, "Collection Stopped", "Mock real-time data collection stopped.")
            self.logger.info("GUI: Mock collection stopped successfully.")
            self._show_tray_message("Collection Stopped", "Mock data collection stopped.")
        else:
            QMessageBox.warning(self, "Stop Error", "Could not stop collection or no collection was running. See logs.")
            self.logger.warning("GUI: Attempt to stop collection failed or no collection was running.")
            self._show_tray_message("Stop Error", "Failed to stop collection or none was running.", icon=QSystemTrayIcon.Icon.Warning)
        self.update_collector_status_display()

    def update_collector_status_display(self):
        status = collector.get_collector_status()
        if status["is_running"]:
            self.collector_status_label.setText(f"Status: Running\nCode: {status['active_internal_code']}\nShoe ID: {status['current_shoe_id']}\nRounds in Shoe: {status['rounds_collected_in_shoe']}")
            self.start_collection_button.setEnabled(False)
            self.stop_collection_button.setEnabled(True)
            self.internal_code_input.setEnabled(False)
            self.internal_code_input.setText(status['active_internal_code'] or "")
        else:
            self.collector_status_label.setText("Status: Idle. Enter code to start.")
            self.start_collection_button.setEnabled(True)
            self.stop_collection_button.setEnabled(False)
            self.internal_code_input.setEnabled(True)
        if status["is_running"]:
             self.load_round_data()
        self._update_auxiliary_info()

    def run_simulation(self):
        self.logger.info("Attempting to run simulation from GUI.")
        selected_shoe_str = self.shoe_id_combo.currentText()
        if not selected_shoe_str or "No shoes" in selected_shoe_str or "Error" in selected_shoe_str:
            self.logger.warning(f"Simulation run attempt failed: Invalid shoe selected ('{selected_shoe_str}').")
            QMessageBox.warning(self, "Simulation Error", "Please select a valid shoe ID for simulation.")
            return
        try: shoe_id_to_simulate = int(selected_shoe_str)
        except ValueError:
            self.logger.warning(f"Simulation run attempt failed: Shoe ID '{selected_shoe_str}' is not a valid integer.")
            QMessageBox.warning(self, "Simulation Error", f"Invalid Shoe ID selected: {selected_shoe_str}")
            return
        selected_strategy_name = self.strategy_combo.currentText()
        strategy_instance = self.strategy_map.get(selected_strategy_name)
        if not strategy_instance:
            self.logger.error(f"Simulation run attempt failed: Strategy '{selected_strategy_name}' not found in strategy_map.")
            QMessageBox.warning(self, "Simulation Error", "Invalid strategy selected.")
            return
        if isinstance(strategy_instance, SimpleMLStrategy) and not strategy_instance.model:
            self.logger.info(f"Strategy '{selected_strategy_name}' is an unfitted SimpleMLStrategy. Performing placeholder fit.")
            if strategy_instance.model is None and hasattr(strategy_instance, '_is_model_fitted'):
                 strategy_instance.model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
            strategy_instance.fit(None, None)
        initial_bankroll = self.initial_bankroll_spin.value()

        bet_sizing_method = self.bet_sizing_combo.currentText().lower()
        fixed_bet_val = self.bet_amount_spin.value()
        percentage_val = self.percentage_bet_spin.value() / 100.0
        kelly_frac_val = self.kelly_fraction_spin.value()
        max_cons_loss_val = self.max_consecutive_losses_spin.value()
        if max_cons_loss_val == 0:
            max_cons_loss_val = None

        stop_loss_perc = self.stop_loss_percentage_spin.value() / 100.0 if self.enable_stop_loss_check.isChecked() else None
        stop_profit_perc = self.stop_profit_percentage_spin.value() / 100.0 if self.enable_stop_profit_check.isChecked() else None

        self.logger.info(f"Starting simulation: Shoe ID {shoe_id_to_simulate}, Strategy '{selected_strategy_name}', "
                         f"Initial Bankroll ${initial_bankroll:.2f}, Bet Sizing: {bet_sizing_method}, "
                         f"FixedAmt: {fixed_bet_val if bet_sizing_method == 'fixed' else 'N/A'}, "
                         f"PercVal: {percentage_val*100 if bet_sizing_method == 'percentage' else 'N/A'}%, "
                         f"KellyFrac: {kelly_frac_val if bet_sizing_method == 'kelly' else 'N/A'}, "
                         f"MaxConsLoss: {max_cons_loss_val if max_cons_loss_val is not None else 'N/A'}, "
                         f"SL%: {stop_loss_perc*100 if stop_loss_perc else 'N/A'}%, "
                         f"SP%: {stop_profit_perc*100 if stop_profit_perc else 'N/A'}%")

        simulator = BettingSimulator(
            strategy=strategy_instance,
            initial_bankroll=initial_bankroll,
            bet_sizing_method=bet_sizing_method,
            fixed_bet_amount=fixed_bet_val,
            percentage_bet_value=percentage_val,
            kelly_fraction=kelly_frac_val,
            stop_loss_percentage=stop_loss_perc,
            stop_profit_percentage=stop_profit_perc,
            max_consecutive_losses=max_cons_loss_val
        )

        final_bankroll = simulator.simulate_shoe(shoe_id=shoe_id_to_simulate)
        profit_loss = final_bankroll - initial_bankroll
        self.result_strategy_label.setText(selected_strategy_name)
        self.result_shoe_id_label.setText(str(shoe_id_to_simulate))
        self.result_initial_bankroll_label.setText(f"{initial_bankroll:.2f} $")
        self.result_final_bankroll_label.setText(f"{final_bankroll:.2f} $")
        self.result_profit_loss_label.setText(f"{profit_loss:.2f} $")
        stop_reason = "Completed all rounds"
        if simulator.stop_loss_threshold is not None and simulator.bankroll <= simulator.stop_loss_threshold + 1e-9 :
             stop_reason = f"Stop-Loss triggered at {simulator.bankroll:.2f}$"
             self._show_tray_message("Simulation Alert", stop_reason, QSystemTrayIcon.Icon.Warning)
        elif simulator.stop_profit_threshold is not None and simulator.bankroll >= simulator.stop_profit_threshold - 1e-9:
             stop_reason = f"Stop-Profit triggered at {simulator.bankroll:.2f}$"
             self._show_tray_message("Simulation Alert", stop_reason, QSystemTrayIcon.Icon.Information)
        elif simulator.max_consecutive_losses is not None and simulator.consecutive_losses >= simulator.max_consecutive_losses :
             stop_reason = f"Max Cons. Losses ({simulator.max_consecutive_losses}) hit"
             self._show_tray_message("Simulation Alert", stop_reason, QSystemTrayIcon.Icon.Warning)
        elif simulator.bankroll < (self.bet_amount_spin.value() if bet_sizing_method == 'fixed' else 1.0) and \
             len(simulator.bet_history) < len(simulator.get_rounds_for_shoe(shoe_id_to_simulate) or []):
             stop_reason = "Insufficient funds"
             self._show_tray_message("Simulation Alert", stop_reason, QSystemTrayIcon.Icon.Warning)
        self.result_stop_reason_label.setText(stop_reason)

        sl_threshold_val_abs = simulator.stop_loss_threshold
        sp_threshold_val_abs = simulator.stop_profit_threshold
        sl_threshold_for_chart = (sl_threshold_val_abs - initial_bankroll) if sl_threshold_val_abs is not None else None
        sp_threshold_for_chart = (sp_threshold_val_abs - initial_bankroll) if sp_threshold_val_abs is not None else None

        self.chart_widget.update_p_l_chart(simulator.bet_history, initial_bankroll,
                                           sl_threshold_for_chart, sp_threshold_for_chart)

        conn = get_db_connection(); cursor = conn.cursor()
        cursor.execute("SELECT round_number_in_shoe, result FROM rounds WHERE shoe_id = ? ORDER BY round_number_in_shoe ASC", (shoe_id_to_simulate,))
        shoe_rounds_data = cursor.fetchall(); conn.close()
        self.chart_widget.update_trend_chart(shoe_rounds_data)

        self.logger.info(f"Simulation complete. Final Bankroll: ${final_bankroll:.2f}, P/L: ${profit_loss:.2f}. Reason: {stop_reason}")
        QMessageBox.information(self, "Simulation Complete",
                                f"Simulation for Shoe ID {shoe_id_to_simulate} with {selected_strategy_name} strategy is complete.\n"
                                f"Initial Bankroll: ${initial_bankroll:.2f}\n"
                                f"Final Bankroll: ${final_bankroll:.2f}\n"
                                f"Profit/Loss: ${profit_loss:.2f}\n"
                                f"Stop Reason: {stop_reason}")

    def _update_bet_sizing_inputs_visibility(self, method_name):
        is_fixed = (method_name == "Fixed")
        is_percentage = (method_name == "Percentage")
        is_kelly = (method_name == "Kelly")
        self.fixed_bet_amount_label.setVisible(is_fixed)
        self.bet_amount_spin.setVisible(is_fixed)
        self.percentage_bet_label.setVisible(is_percentage)
        self.percentage_bet_spin.setVisible(is_percentage)
        self.kelly_fraction_label.setVisible(is_kelly)
        self.kelly_fraction_spin.setVisible(is_kelly)
        if is_fixed:
            self.fixed_bet_amount_label.setText("Fixed Bet Amount:")

    def _setup_auxiliary_area_group(self):
        aux_group = QGroupBox("💡 Auxiliary Info & Suggestions")
        aux_layout = QVBoxLayout()
        self.todays_performance_label = QLabel("Today's Performance: Calculating...")
        aux_layout.addWidget(self.todays_performance_label)
        self.model_commentary_label = QLabel("Model Commentary: N/A")
        aux_layout.addWidget(self.model_commentary_label)
        self.intelligent_suggestion_text = QTextEdit()
        self.intelligent_suggestion_text.setReadOnly(True)
        self.intelligent_suggestion_text.setPlaceholderText("Suggestions will appear here based on live data/strategy...")
        self.intelligent_suggestion_text.setFixedHeight(80)
        aux_layout.addWidget(QLabel("Intelligent Suggestions:"))
        aux_layout.addWidget(self.intelligent_suggestion_text)
        aux_group.setLayout(aux_layout)
        self.left_panel_layout.addWidget(aux_group)

    def _update_auxiliary_info(self):
        total_pl, total_bets = get_todays_performance_summary()
        self.todays_performance_label.setText(f"Today's Sim P/L: ${total_pl:.2f} ({total_bets} bets)")

        current_strategy_name = self.strategy_combo.currentText()
        if "Random" in current_strategy_name or "ML" in current_strategy_name:
            self.model_commentary_label.setText("Model Commentary: Current strategy may involve higher variance.")
        elif total_pl < -50 :
             self.model_commentary_label.setText("Model Commentary: Caution - significant recent sim losses.")
        else:
            self.model_commentary_label.setText("Model Commentary: Standard operation.")

        collector_status = collector.get_collector_status()
        if collector_status["is_running"] and collector_status["current_shoe_id"] is not None:
            conn = get_db_connection(); cursor = conn.cursor()
            cursor.execute("SELECT result, round_number_in_shoe FROM rounds WHERE shoe_id = ? ORDER BY round_number_in_shoe DESC LIMIT 1", (collector_status["current_shoe_id"],))
            last_round_row = cursor.fetchone(); conn.close()

            # Update Real Bet Panel info
            self.rb_shoe_id_label.setText(str(collector_status["current_shoe_id"]))
            next_round_num_for_bet = collector_status["rounds_collected_in_shoe"] + 1
            self.rb_round_number_label.setText(str(next_round_num_for_bet))

            if last_round_row:
                suggestion = f"Collector Active. Last round ({last_round_row['round_number_in_shoe']}) in shoe {collector_status['current_shoe_id']} was: {last_round_row['result']}. Consider this trend."
                self.intelligent_suggestion_text.setText(suggestion)
            else:
                self.intelligent_suggestion_text.setText(f"Collector Active (Shoe {collector_status['current_shoe_id']}). Waiting for first round data...")
        else:
            self.intelligent_suggestion_text.setText("Collector idle. No live suggestions.")
            self.rb_shoe_id_label.setText("N/A (Collector Idle)")
            self.rb_round_number_label.setText("N/A")

    def place_simulated_real_bet(self):
        self.logger.info("GUI: 'Place Simulated Real Bet' button clicked.")
        collector_status = collector.get_collector_status()
        if not collector_status["is_running"] or collector_status["current_shoe_id"] is None:
            QMessageBox.warning(self, "Bet Error", "Real-time collector is not running or no active shoe.")
            self.real_bet_status_label.setText("Status: Error - Collector not active.")
            return

        # For a "real" bet, we'd typically bet on the *next* outcome of the *current* live round.
        # The round_id in the 'bets' table should correspond to the round whose outcome determines the bet's win/loss.
        # If collector just finished round N, this bet is for round N+1.
        # We need to ensure a round record for N+1 exists or can be associated.
        # For simplicity with mock collector: assume the bet is for the *next round to be collected*.
        # This means the 'round_id' might not exist yet when place_bet_sync is called.
        # The `bet_manager` currently requires `round_id_from_collector`.
        # This implies a design choice:
        # 1. Create a placeholder "pending" round in `rounds` table for the upcoming bet.
        # 2. Or, modify `bets` table to allow `round_id` to be NULL initially, and update it later.
        # 3. Or, associate bet with `shoe_id` and `next_round_number`.
        # For now, let's assume we need a valid round_id. This means we can only place 'real' bets on *already recorded* rounds
        # if we interpret `round_id_from_collector` strictly.
        # This isn't how real betting works.
        #
        # Let's adjust: The bet is for the *current live context*. The `round_id` in `bets` table
        # will refer to the round whose *outcome* settles this bet.
        # If the collector just reported round `N`, a bet placed now is for round `N`.
        # So, `round_id_from_collector` should be the ID of the *currently forming or just completed* round.

        # Get the latest round_id from the current shoe to associate the bet.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT round_id FROM rounds WHERE shoe_id = ? ORDER BY round_number_in_shoe DESC LIMIT 1",
                       (collector_status["current_shoe_id"],))
        last_round_record = cursor.fetchone()
        conn.close()

        if not last_round_record:
            QMessageBox.warning(self, "Bet Error", "No rounds recorded yet for the current shoe. Cannot place bet.")
            self.real_bet_status_label.setText("Status: Error - No rounds in current shoe.")
            return

        target_round_id = last_round_record['round_id'] # Bet is associated with the most recent (or current live) round

        bet_on = self.rb_bet_on_combo.currentText()
        amount = self.rb_bet_amount_spin.value()
        strategy_name = self.rb_strategy_label.text() # Could be dynamic later

        self.real_bet_status_label.setText(f"Status: Placing bet on {bet_on} for ${amount:.2f}...")
        QApplication.processEvents() # Update GUI before blocking call

        response = self.bet_manager.place_bet_sync(
            round_id_from_collector=target_round_id,
            bet_on=bet_on,
            amount=amount,
            strategy_name=strategy_name
        )

        if response['success']:
            self.real_bet_status_label.setText(f"Status: Bet ID {response['bet_id']} placed successfully!")
            self._show_tray_message("Real Bet (Simulated)", f"Bet on {bet_on} for ${amount:.2f} placed.")
        else:
            self.real_bet_status_label.setText(f"Status: Bet failed - {response['message']}")
            self._show_tray_message("Real Bet Failed (Simulated)", response['message'], icon=QSystemTrayIcon.Icon.Warning)

        self._update_auxiliary_info() # Refresh today's performance which includes real bets if they had payout


    def _setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = os.path.join(PROJECT_ROOT, "assets", "app_icon.png")
        if not os.path.exists(icon_path) or os.path.getsize(icon_path) == 0: # Check if placeholder or missing
            self.logger.warning(f"Tray icon {icon_path} not found or empty. Using default window icon.")
            self.tray_icon.setIcon(self.windowIcon())
        else:
            self.tray_icon.setIcon(QIcon(icon_path))
        self.tray_icon.setToolTip("XiaoXiao1 Baccarat Assistant")
        show_action = QAction("Show", self); quit_action = QAction("Exit", self)
        show_action.triggered.connect(self.showNormal)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu = QMenu(); tray_menu.addAction(show_action); tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.logger.info("System tray icon created.")
        self.tray_icon.activated.connect(self._tray_icon_activated)

    def _tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isMinimized() or not self.isVisible():
                self.showNormal()
            self.activateWindow()

    def _show_tray_message(self, title, message, icon=QSystemTrayIcon.Icon.Information, duration_ms=5000):
        if QSystemTrayIcon.isSystemTrayAvailable() and self.tray_icon.isVisible():
            self.tray_icon.showMessage(title, message, icon, duration_ms)
            self.logger.info(f"Tray Message: Title='{title}', Msg='{message}'")
        else:
            self.logger.info(f"System tray not available or icon not visible. Message not shown: Title='{title}', Msg='{message}'")

    def closeEvent(self, event):
        self.logger.info("Close event triggered. Stopping real-time collection if running.")
        if collector.get_collector_status()["is_running"]:
            self.stop_realtime_collection()
        super().closeEvent(event)

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys
    try:
        from core.utils import create_tables, app_logger as main_app_logger
        main_app_logger.info("MainWindow.py executed directly. Initializing tables...")
        create_tables()
        main_app_logger.info("Tables initialized (if not already existing).")
    except Exception as e:
        print(f"Critical error during pre-GUI setup (tables/logger): {e}")
        if 'main_app_logger' in locals(): main_app_logger.critical(f"Could not ensure tables are created: {e}", exc_info=True)
        sys.exit(1)

    app = QApplication(sys.argv)
    main_app_logger.info("QApplication initialized. Creating MainWindow...")

    qt_handler_for_direct_run = None
    # Check if running within the context of run_gui.py which might have set this up
    # This check is a bit fragile and depends on execution context.
    if hasattr(sys.modules.get('__main__'), 'global_qt_log_handler'):
         qt_handler_for_direct_run = sys.modules['__main__'].global_qt_log_handler
    else:
        try:
            from gui.widgets.log_display_widget import QtLogHandler
            import logging
            qt_handler_for_direct_run = QtLogHandler()
            gui_formatter = logging.Formatter('%(asctime)s [%(levelname)-7s] %(message)s (%(module)s:%(lineno)d)')
            qt_handler_for_direct_run.setFormatter(gui_formatter)
            qt_handler_for_direct_run.setLevel(logging.DEBUG)
            app_logger.addHandler(qt_handler_for_direct_run)
            main_app_logger.info("Direct run of MainWindow: Created a new QtLogHandler for GUI display.")
        except Exception as e_direct_log:
            main_app_logger.error(f"Direct run of MainWindow: Failed to create test QtLogHandler: {e_direct_log}")

    window = MainWindow(qt_log_handler=qt_handler_for_direct_run)

    # --- Hot Reload Group Setup ---
    def _setup_hot_reload_group(self):
        hot_reload_group = QGroupBox("♻️ Hot Reload")
        hr_layout = QVBoxLayout() # Use QVBoxLayout for simple button list

        self.reload_strats_button = QPushButton("Reload Strategy Configs & Models")
        self.reload_strats_button.clicked.connect(self.handle_hot_reload_strategies)
        hr_layout.addWidget(self.reload_strats_button)

        # Add more specific reload buttons if needed later, e.g., only ML models, only specific JSON

        hot_reload_group.setLayout(hr_layout)
        self.left_panel_layout.addWidget(hot_reload_group)

    def handle_hot_reload_strategies(self):
        self.logger.info("Hot reload triggered from GUI.")

        # Track if any strategy name changed to update combo box properly
        strategy_names_before_reload = set(self.strategy_map.keys())

        reloaded_strategies = {}
        failed_reloads = []

        for name, strat_instance in self.strategy_map.items():
            reloaded = False
            new_instance = None
            if isinstance(strat_instance, StrategyCombiner):
                self.logger.info(f"Attempting to hot reload StrategyCombiner: {name}")
                # Re-create the instance to ensure it picks up file changes correctly
                # This assumes config_filename is an attribute we can access.
                # If StrategyCombiner's reload_config is robust enough to rename itself,
                # we might not need to re-create, but re-creation is safer for complex changes.
                try:
                    config_file = strat_instance.config_filename
                    new_instance = StrategyCombiner(config_filename=config_file)
                    # "Train" fallback ML if needed (as in _update_strategy_map)
                    if new_instance.fallback_strategy and isinstance(new_instance.fallback_strategy, SimpleMLStrategy):
                        if not new_instance.fallback_strategy.model or not hasattr(new_instance.fallback_strategy.model, 'classes_'):
                             new_instance.fallback_strategy.fit(None,None)
                    reloaded_strategies[new_instance.strategy_name] = new_instance # Use new name as key
                    reloaded = True
                    self.logger.info(f"Successfully reloaded and re-initialized {name} as {new_instance.strategy_name}.")
                except Exception as e:
                    self.logger.error(f"Failed to re-initialize StrategyCombiner {name}: {e}", exc_info=True)
                    failed_reloads.append(name)
                    reloaded_strategies[name] = strat_instance # Keep old instance on failure

            elif isinstance(strat_instance, SklearnStyleStrategy): # Includes SimpleMLStrategy
                self.logger.info(f"Attempting to hot reload ML Model for strategy: {name} from {strat_instance.model_path}")
                try:
                    strat_instance.load_model() # Reloads from its stored model_path
                    # If load_model doesn't re-init a default model on failure, we might need to check self.model
                    if strat_instance.model is None and isinstance(strat_instance, SimpleMLStrategy): # If load failed and it's SimpleML
                        self.logger.warning(f"Model for {name} was None after load_model, re-initializing default RandomForest.")
                        strat_instance.model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')

                    reloaded_strategies[name] = strat_instance # Keep same instance, model reloaded
                    reloaded = True
                except Exception as e:
                    self.logger.error(f"Failed to reload model for {name}: {e}", exc_info=True)
                    failed_reloads.append(name)
                    reloaded_strategies[name] = strat_instance # Keep old instance

            else: # For other types of strategies not covered by specific reload logic
                reloaded_strategies[name] = strat_instance


        # Update the main strategy map
        self.strategy_map = reloaded_strategies

        # Refresh the strategy ComboBox in the simulation panel
        # This is important if names changed or to reflect reloaded state (though state isn't visible in name)
        current_sim_strategy_text = self.strategy_combo.currentText()
        self.strategy_combo.clear()
        self.strategy_combo.addItems(self.strategy_map.keys())
        if current_sim_strategy_text in self.strategy_map: # If old selection still valid by name
            self.strategy_combo.setCurrentText(current_sim_strategy_text)
        elif self.strategy_map: # Select first available if old one gone
             self.strategy_combo.setCurrentIndex(0)

        # Also update live strategy combo if it exists and uses the same map
        # if hasattr(self, 'live_strategy_combo'): ... similar logic ...

        if not failed_reloads:
            QMessageBox.information(self, "Hot Reload", "Strategies and ML Models reloaded successfully (if applicable).")
            self.logger.info("All applicable strategies/models reloaded successfully.")
        else:
            QMessageBox.warning(self, "Hot Reload", f"Some strategies/models failed to reload: {', '.join(failed_reloads)}. Check logs.")
            self.logger.warning(f"Strategies/models that failed to reload: {failed_reloads}")

        self._update_auxiliary_info() # Refresh commentary which might depend on strategy
    main_app_logger.info("MainWindow created. Showing window...")
    window.show()
    exit_code = app.exec()
    main_app_logger.info(f"Application exiting with code {exit_code}.")
    sys.exit(exit_code)
