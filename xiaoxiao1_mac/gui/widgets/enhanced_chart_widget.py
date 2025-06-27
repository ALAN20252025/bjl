import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QCheckBox, QPushButton, QComboBox, QLabel, QSpinBox
from PySide6.QtCore import Qt, Signal
from core.utils import app_logger, get_db_connection
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

class EnhancedChartWidget(QWidget):
    """增强版图表组件，提供更多分析功能"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = app_logger
        self.logger.info("EnhancedChartWidget initializing.")
        
        self.setup_ui()
        self.current_bet_history = []
        self.current_initial_bankroll = 0
        self.current_rounds_data = []
        
    def setup_ui(self):
        """设置UI布局"""
        main_layout = QVBoxLayout(self)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 图表区域
        chart_splitter = QSplitter(Qt.Vertical)
        
        # P&L 图表（上部）
        self.pl_chart_widget = self.create_pl_chart()
        chart_splitter.addWidget(self.pl_chart_widget)
        
        # 多功能图表区域（下部）
        multi_chart_splitter = QSplitter(Qt.Horizontal)
        
        # 趋势图表
        self.trend_chart_widget = self.create_trend_chart()
        multi_chart_splitter.addWidget(self.trend_chart_widget)
        
        # 统计图表
        self.stats_chart_widget = self.create_stats_chart()
        multi_chart_splitter.addWidget(self.stats_chart_widget)
        
        chart_splitter.addWidget(multi_chart_splitter)
        chart_splitter.setSizes([300, 250])
        multi_chart_splitter.setSizes([300, 300])
        
        main_layout.addWidget(chart_splitter)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("图表控制")
        layout = QHBoxLayout(panel)
        
        # P&L图表选项
        self.show_sl_sp_check = QCheckBox("显示止损止盈线")
        self.show_sl_sp_check.setChecked(True)
        layout.addWidget(self.show_sl_sp_check)
        
        # 趋势图表选项
        layout.addWidget(QLabel("趋势显示:"))
        self.trend_mode_combo = QComboBox()
        self.trend_mode_combo.addItems(["结果点", "连胜连败", "路珠图"])
        layout.addWidget(self.trend_mode_combo)
        
        # 统计图表选项
        layout.addWidget(QLabel("统计类型:"))
        self.stats_mode_combo = QComboBox()
        self.stats_mode_combo.addItems(["胜率分布", "连胜连败统计", "投注分析", "风险指标"])
        layout.addWidget(self.stats_mode_combo)
        
        # 数据范围
        layout.addWidget(QLabel("显示范围:"))
        self.range_spin = QSpinBox()
        self.range_spin.setRange(10, 1000)
        self.range_spin.setValue(100)
        self.range_spin.setSuffix("局")
        layout.addWidget(self.range_spin)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新图表")
        refresh_btn.clicked.connect(self.refresh_all_charts)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        # 连接信号
        self.show_sl_sp_check.toggled.connect(self.update_p_l_chart)
        self.trend_mode_combo.currentTextChanged.connect(self.update_trend_chart)
        self.stats_mode_combo.currentTextChanged.connect(self.update_stats_chart)
        
        return panel
        
    def create_pl_chart(self):
        """创建P&L图表"""
        widget = pg.PlotWidget()
        widget.setBackground('w')
        widget.setTitle("资金曲线 (P&L)", color="k", size="14pt")
        widget.setLabel('left', '盈亏金额 ($)', color='k')
        widget.setLabel('bottom', '投注序号', color='k')
        widget.showGrid(x=True, y=True)
        widget.addLegend()
        
        # P&L曲线
        self.pl_curve = widget.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolSize=4, name="P&L")
        
        # 移动平均线
        self.ma_curve = widget.plot(pen=pg.mkPen('orange', width=2, style=Qt.DashLine), name="移动平均")
        
        # 止损止盈线
        self.sl_line = pg.InfiniteLine(angle=0, movable=False, 
                                      pen=pg.mkPen('r', width=2, style=Qt.DashLine), 
                                      label='止损线')
        self.sp_line = pg.InfiniteLine(angle=0, movable=False, 
                                      pen=pg.mkPen('g', width=2, style=Qt.DashLine), 
                                      label='止盈线')
        
        widget.addItem(self.sl_line)
        widget.addItem(self.sp_line)
        
        # 风险区域
        self.risk_fill = pg.FillBetweenItem(brush=pg.mkBrush(255, 0, 0, 30))
        self.profit_fill = pg.FillBetweenItem(brush=pg.mkBrush(0, 255, 0, 30))
        
        return widget
        
    def create_trend_chart(self):
        """创建趋势图表"""
        widget = pg.PlotWidget()
        widget.setBackground('w')
        widget.setTitle("游戏趋势分析", color="k", size="14pt")
        widget.setLabel('left', '结果', color='k')
        widget.setLabel('bottom', '局数', color='k')
        widget.showGrid(x=True, y=True)
        
        # 设置Y轴标签
        y_axis = widget.getAxis('left')
        y_axis.setTicks([[(1, 'P'), (2, 'B'), (3, 'T')]])
        
        self.trend_scatter = pg.ScatterPlotItem(size=8)
        widget.addItem(self.trend_scatter)
        
        return widget
        
    def create_stats_chart(self):
        """创建统计图表"""
        widget = pg.PlotWidget()
        widget.setBackground('w')
        widget.setTitle("统计分析", color="k", size="14pt")
        widget.showGrid(x=True, y=True)
        
        return widget
        
    def update_p_l_chart(self):
        """更新P&L图表"""
        if not self.current_bet_history:
            self.pl_curve.setData([], [])
            self.ma_curve.setData([], [])
            return
            
        # 计算数据
        bet_numbers = list(range(len(self.current_bet_history) + 1))
        pl_values = [0]  # 初始值
        
        for i, record in enumerate(self.current_bet_history):
            pl = record[5] - self.current_initial_bankroll  # record[5] 是新的bankroll
            pl_values.append(pl)
            
        # 更新主曲线
        self.pl_curve.setData(bet_numbers, pl_values)
        
        # 计算移动平均（窗口大小为10）
        if len(pl_values) > 10:
            ma_values = []
            for i in range(len(pl_values)):
                start_idx = max(0, i - 9)
                ma_values.append(np.mean(pl_values[start_idx:i+1]))
            self.ma_curve.setData(bet_numbers, ma_values)
        
        # 更新止损止盈线
        if self.show_sl_sp_check.isChecked():
            # 假设止损20%，止盈50%
            stop_loss = -self.current_initial_bankroll * 0.2
            stop_profit = self.current_initial_bankroll * 0.5
            
            self.sl_line.setPos(stop_loss)
            self.sp_line.setPos(stop_profit)
            self.sl_line.show()
            self.sp_line.show()
        else:
            self.sl_line.hide()
            self.sp_line.hide()
            
        self.pl_chart_widget.autoRange()
        
    def update_trend_chart(self):
        """更新趋势图表"""
        mode = self.trend_mode_combo.currentText()
        
        if mode == "结果点":
            self.update_result_scatter()
        elif mode == "连胜连败":
            self.update_streak_chart()
        elif mode == "路珠图":
            self.update_bead_road_chart()
            
    def update_result_scatter(self):
        """更新结果散点图"""
        if not self.current_rounds_data:
            return
            
        spots = []
        result_map = {'Player': (1, 'blue'), 'Banker': (2, 'red'), 'Tie': (3, 'green')}
        
        for r_data in self.current_rounds_data[-self.range_spin.value():]:
            round_num = r_data['round_number_in_shoe']
            result = r_data['result']
            y_val, color = result_map.get(result, (0, 'black'))
            spots.append({'pos': (round_num, y_val), 'brush': pg.mkBrush(color)})
            
        self.trend_scatter.setData(spots)
        self.trend_chart_widget.autoRange()
        
    def update_streak_chart(self):
        """更新连胜连败图表"""
        self.trend_chart_widget.clear()
        self.trend_chart_widget.setTitle("连胜连败分析", color="k", size="14pt")
        self.trend_chart_widget.setLabel('left', '连续次数', color='k')
        self.trend_chart_widget.setLabel('bottom', '序号', color='k')
        
        if not self.current_rounds_data:
            return
            
        # 计算连胜连败
        streaks = self.calculate_streaks()
        
        x_data = list(range(len(streaks)))
        y_data = [abs(s) for s in streaks]
        colors = ['green' if s > 0 else 'red' for s in streaks]
        
        # 绘制条形图
        bargraph = pg.BarGraphItem(x=x_data, height=y_data, width=0.8, 
                                  brushes=colors, pens=colors)
        self.trend_chart_widget.addItem(bargraph)
        
    def update_bead_road_chart(self):
        """更新路珠图"""
        self.trend_chart_widget.clear()
        self.trend_chart_widget.setTitle("路珠图", color="k", size="14pt")
        
        if not self.current_rounds_data:
            return
            
        # 简化的路珠图实现
        spots = []
        colors = {'Player': 'blue', 'Banker': 'red', 'Tie': 'green'}
        
        for i, r_data in enumerate(self.current_rounds_data[-50:]):  # 显示最近50局
            x = i % 10  # 10列
            y = i // 10  # 行数
            result = r_data['result']
            color = colors.get(result, 'black')
            spots.append({'pos': (x, y), 'brush': pg.mkBrush(color), 'size': 12})
            
        scatter = pg.ScatterPlotItem()
        scatter.setData(spots)
        self.trend_chart_widget.addItem(scatter)
        
    def update_stats_chart(self):
        """更新统计图表"""
        mode = self.stats_mode_combo.currentText()
        
        if mode == "胜率分布":
            self.update_win_rate_chart()
        elif mode == "连胜连败统计":
            self.update_streak_stats_chart()
        elif mode == "投注分析":
            self.update_bet_analysis_chart()
        elif mode == "风险指标":
            self.update_risk_metrics_chart()
            
    def update_win_rate_chart(self):
        """更新胜率分布图表"""
        self.stats_chart_widget.clear()
        self.stats_chart_widget.setTitle("胜率分布", color="k", size="14pt")
        
        if not self.current_bet_history:
            return
            
        # 计算胜率
        wins = sum(1 for record in self.current_bet_history if record[4] > 0)  # record[4] 是payout
        total = len(self.current_bet_history)
        win_rate = wins / total if total > 0 else 0
        
        # 绘制饼图（简化版）
        labels = ['胜', '负']
        values = [wins, total - wins]
        colors = ['green', 'red']
        
        angles = [v / total * 360 for v in values]
        
        # 使用条形图代替饼图
        bargraph = pg.BarGraphItem(x=[0, 1], height=values, width=0.6, 
                                  brushes=colors, pens=colors)
        self.stats_chart_widget.addItem(bargraph)
        
        # 设置轴标签
        self.stats_chart_widget.setLabel('bottom', '结果类型')
        self.stats_chart_widget.setLabel('left', '次数')
        
    def update_streak_stats_chart(self):
        """更新连胜连败统计图表"""
        self.stats_chart_widget.clear()
        self.stats_chart_widget.setTitle("连胜连败统计", color="k", size="14pt")
        
        if not self.current_rounds_data:
            return
            
        streaks = self.calculate_streaks()
        
        # 统计连胜连败分布
        win_streaks = [s for s in streaks if s > 0]
        lose_streaks = [abs(s) for s in streaks if s < 0]
        
        # 绘制直方图
        if win_streaks:
            win_hist, win_bins = np.histogram(win_streaks, bins=10)
            win_curve = self.stats_chart_widget.plot(win_bins[:-1], win_hist, 
                                                    pen='g', stepMode=True, 
                                                    fillLevel=0, brush=(0, 255, 0, 100),
                                                    name='连胜')
            
        if lose_streaks:
            lose_hist, lose_bins = np.histogram(lose_streaks, bins=10)
            lose_curve = self.stats_chart_widget.plot(lose_bins[:-1], lose_hist, 
                                                     pen='r', stepMode=True, 
                                                     fillLevel=0, brush=(255, 0, 0, 100),
                                                     name='连败')
        
        self.stats_chart_widget.addLegend()
        
    def update_bet_analysis_chart(self):
        """更新投注分析图表"""
        self.stats_chart_widget.clear()
        self.stats_chart_widget.setTitle("投注分析", color="k", size="14pt")
        
        if not self.current_bet_history:
            return
            
        # 分析投注额分布
        bet_amounts = [record[2] for record in self.current_bet_history]  # record[2] 是bet_amount
        
        if bet_amounts:
            hist, bins = np.histogram(bet_amounts, bins=20)
            curve = self.stats_chart_widget.plot(bins[:-1], hist, 
                                                pen='b', stepMode=True, 
                                                fillLevel=0, brush=(0, 0, 255, 100))
            
        self.stats_chart_widget.setLabel('bottom', '投注金额')
        self.stats_chart_widget.setLabel('left', '频次')
        
    def update_risk_metrics_chart(self):
        """更新风险指标图表"""
        self.stats_chart_widget.clear()
        self.stats_chart_widget.setTitle("风险指标", color="k", size="14pt")
        
        if not self.current_bet_history:
            return
            
        # 计算风险指标
        pl_values = []
        for record in self.current_bet_history:
            pl = record[5] - self.current_initial_bankroll
            pl_values.append(pl)
            
        if pl_values:
            # 计算回撤
            running_max = np.maximum.accumulate(pl_values)
            drawdown = running_max - pl_values
            
            # 绘制回撤曲线
            self.stats_chart_widget.plot(range(len(drawdown)), drawdown, 
                                       pen='r', name='回撤')
            
            # 计算并显示关键指标
            max_drawdown = np.max(drawdown)
            self.logger.info(f"最大回撤: {max_drawdown:.2f}")
            
        self.stats_chart_widget.setLabel('bottom', '交易序号')
        self.stats_chart_widget.setLabel('left', '回撤金额')
        
    def calculate_streaks(self):
        """计算连胜连败"""
        if not self.current_bet_history:
            return []
            
        streaks = []
        current_streak = 0
        
        for record in self.current_bet_history:
            payout = record[4]  # record[4] 是payout
            
            if payout > 0:  # 赢
                if current_streak >= 0:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
            else:  # 输
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    streaks.append(current_streak)
                    current_streak = -1
                    
        if current_streak != 0:
            streaks.append(current_streak)
            
        return streaks
        
    def update_data(self, bet_history, initial_bankroll, rounds_data=None):
        """更新数据"""
        self.current_bet_history = bet_history
        self.current_initial_bankroll = initial_bankroll
        if rounds_data:
            self.current_rounds_data = rounds_data
            
        self.refresh_all_charts()
        
    def refresh_all_charts(self):
        """刷新所有图表"""
        self.update_p_l_chart()
        self.update_trend_chart()
        self.update_stats_chart()
        
    def clear_charts(self):
        """清空所有图表"""
        self.pl_curve.setData([], [])
        self.ma_curve.setData([], [])
        self.trend_chart_widget.clear()
        self.stats_chart_widget.clear()
        self.sl_line.hide()
        self.sp_line.hide() 