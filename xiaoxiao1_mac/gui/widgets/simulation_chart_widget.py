import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt
from core.utils import app_logger

class SimulationChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = app_logger
        self.logger.info("SimulationChartWidget initializing.")

        self.main_layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)

        # P/L Plot Widget
        self.pl_plot_widget = pg.PlotWidget()
        self.pl_plot_widget.setBackground('w')
        self.pl_plot_widget.setTitle("Simulation P/L Over Rounds", color="k", size="12pt")
        self.pl_plot_widget.setLabel('left', 'Profit/Loss ($)', color='k')
        self.pl_plot_widget.setLabel('bottom', 'Round Number (Bet Sequence)', color='k')
        self.pl_plot_widget.showGrid(x=True, y=True)
        self.pl_plot_widget.addLegend()

        self.p_l_curve = self.pl_plot_widget.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolPen='b', symbolBrush=0.2, name="P/L")

        # Lines for Stop-Loss and Stop-Profit
        self.sl_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1, style=Qt.DashLine), label='Stop-Loss', labelOpts={'position':0.1, 'color': (200,0,0), 'movable': True})
        self.sp_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=1, style=Qt.DashLine), label='Stop-Profit', labelOpts={'position':0.1, 'color': (0,200,0), 'movable': True})
        self.sl_line.hide() # Initially hidden
        self.sp_line.hide() # Initially hidden
        self.pl_plot_widget.addItem(self.sl_line)
        self.pl_plot_widget.addItem(self.sp_line)

        self.splitter.addWidget(self.pl_plot_widget)

        # Basic Trend Plot Widget (Sequential Results)
        self.trend_plot_widget = pg.PlotWidget()
        self.trend_plot_widget.setBackground('w')
        self.trend_plot_widget.setTitle("Round Results Trend", color="k", size="12pt")
        self.trend_plot_widget.setLabel('left', 'Result (P=1, B=2, T=3)', color='k') # Arbitrary Y values for distinct plotting
        self.trend_plot_widget.setLabel('bottom', 'Round Number in Shoe', color='k')
        self.trend_plot_widget.showGrid(x=True, y=True)
        # Set Y-axis ticks for clarity if using discrete values
        y_axis = self.trend_plot_widget.getAxis('left')
        y_axis.setTicks([[(1, 'Player'), (2, 'Banker'), (3, 'Tie')]])

        self.trend_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        self.trend_plot_widget.addItem(self.trend_scatter)
        self.splitter.addWidget(self.trend_plot_widget)

        self.splitter.setSizes([200, 150]) # Initial relative sizes

        self.logger.info("SimulationChartWidget initialized successfully with P/L and Trend plots.")

    def update_p_l_chart(self, bet_history, initial_bankroll):
        self.logger.info(f"Updating P/L chart with {len(bet_history)} data points. Initial bankroll: {initial_bankroll}")
        if not bet_history:
            self.p_l_curve.setData([], [])
            self.logger.info("P/L Chart cleared due to empty bet_history.")
            return

        round_numbers = [0] # x-axis: sequence of bets
        profit_loss_values = [0] # y-axis: P/L

        for i, record in enumerate(bet_history):
            # record = (round_id, bet_on, bet_amount, actual_result, payout, new_bankroll_after_round)
            round_numbers.append(i + 1) # X-axis is the bet number in the sequence
            p_l = record[5] - initial_bankroll
            profit_loss_values.append(p_l)

        self.logger.debug(f"P/L Chart data - Bet Sequence: {round_numbers}, P/L: {profit_loss_values}")
        self.p_l_curve.setData(round_numbers, profit_loss_values)
        self.pl_plot_widget.autoRange()
        self.logger.info("P/L Chart updated successfully.")

    def update_p_l_chart(self, bet_history, initial_bankroll, stop_loss_pl=None, stop_profit_pl=None): # Added new params
        self.logger.info(f"Updating P/L chart. Bets: {len(bet_history)}, Initial Bankroll: {initial_bankroll}, SL P/L: {stop_loss_pl}, SP P/L: {stop_profit_pl}")
        if not bet_history:
            self.p_l_curve.setData([], [])
            self.sl_line.hide()
            self.sp_line.hide()
            self.logger.info("P/L Chart cleared (empty history).")
            return

        round_numbers = [0]
        profit_loss_values = [0]

        for i, record in enumerate(bet_history):
            round_numbers.append(i + 1)
            p_l = record[5] - initial_bankroll
            profit_loss_values.append(p_l)

        self.logger.debug(f"P/L Chart data - Bet Sequence: {round_numbers}, P/L: {profit_loss_values}")
        self.p_l_curve.setData(round_numbers, profit_loss_values)

        if stop_loss_pl is not None:
            self.sl_line.setPos(stop_loss_pl)
            self.sl_line.show()
        else:
            self.sl_line.hide()

        if stop_profit_pl is not None:
            self.sp_line.setPos(stop_profit_pl)
            self.sp_line.show()
        else:
            self.sp_line.hide()

        self.pl_plot_widget.autoRange()
        self.logger.info("P/L Chart updated successfully with SL/SP lines.")


    def update_trend_chart(self, rounds_data):
        """
        Updates the trend chart with results from rounds_data.
        Args:
            rounds_data: A list of round data, typically fetched for a shoe.
                         Each item should be a dict-like object with 'round_number_in_shoe' and 'result'.
        """
        self.logger.info(f"Updating Trend chart with {len(rounds_data)} rounds.")
        if not rounds_data:
            self.trend_scatter.setData([], [])
            self.logger.info("Trend Chart cleared due to empty rounds_data.")
            return

        spots = []
        # Mapping results to Y values and colors for the scatter plot
        result_map = {'Player': (1, 'blue'), 'Banker': (2, 'red'), 'Tie': (3, 'green')}

        for r_data in rounds_data:
            round_num = r_data['round_number_in_shoe']
            result = r_data['result']
            y_val, color = result_map.get(result, (0, 'black')) # Default for unknown
            spots.append({'pos': (round_num, y_val), 'brush': pg.mkBrush(color), 'symbol': 'o'})

        self.logger.debug(f"Trend Chart data spots: {len(spots)}")
        self.trend_scatter.setData(spots)
        self.trend_plot_widget.autoRange()
        self.logger.info("Trend Chart updated successfully.")

    def clear_charts(self):
        self.p_l_curve.setData([], [])
        self.trend_scatter.setData([], []) # Clear trend scatter plot too
        self.logger.info("All simulation charts cleared.")
        self.sl_line.hide() # Ensure lines are hidden on clear
        self.sp_line.hide()


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys
    # Fallback basic logger if app_logger isn't available (e.g. direct run without full setup)
    try:
        from core.utils import app_logger # Adjusted for direct run if project root in path
    except ImportError:
        # Try one level up for core if running from widgets directory
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        try:
            from core.utils import app_logger
        except ImportError:
            import logging
            app_logger = logging.getLogger("FallbackChartLogger")
            app_logger.setLevel(logging.DEBUG)
            if not app_logger.hasHandlers():
                ch = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                app_logger.addHandler(ch)
            app_logger.info("Using fallback logger for SimulationChartWidget direct test.")


    app = QApplication(sys.argv)
    chart_view = SimulationChartWidget()

    # Example P/L data
    test_initial_bankroll = 1000
    test_bet_history = [
        (1, 'Player', 10, 'Player', 10, 1010), (2, 'Banker', 10, 'Player', -10, 1000),
        (3, 'Player', 10, 'Banker', -10, 990), (4, 'Tie', 5, 'Tie', 40, 1030),
        (5, 'Banker', 20, 'Banker', 19, 1049), (6, 'Player', 10, 'Banker', -10, 1039),
    ]
    chart_view.update_p_l_chart(test_bet_history, test_initial_bankroll)

    # Example Trend data (simulating rounds from a shoe)
    test_rounds_data = [
        {'round_number_in_shoe': 1, 'result': 'Player'}, {'round_number_in_shoe': 2, 'result': 'Banker'},
        {'round_number_in_shoe': 3, 'result': 'Banker'}, {'round_number_in_shoe': 4, 'result': 'Tie'},
        {'round_number_in_shoe': 5, 'result': 'Player'}, {'round_number_in_shoe': 6, 'result': 'Player'},
        {'round_number_in_shoe': 7, 'result': 'Banker'},
    ]
    chart_view.update_trend_chart(test_rounds_data)

    chart_view.setWindowTitle("Test P/L and Trend Charts")
    chart_view.resize(800, 600)
    chart_view.show()
    sys.exit(app.exec())
