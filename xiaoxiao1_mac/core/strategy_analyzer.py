"""
策略分析器 - 提供深度策略分析功能
包括性能评估、风险分析、参数优化等
"""

import sqlite3
import json
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

from .utils import get_db_connection, app_logger
from .bet_simulator import BettingSimulator
from .strategy_rules import BaseStrategy


class StrategyAnalyzer:
    """策略分析器"""
    
    def __init__(self):
        self.logger = app_logger
        self.simulator = BettingSimulator()
        
    def analyze_strategy_performance(self, strategy: BaseStrategy, 
                                   shoe_ids: List[int] = None,
                                   initial_bankroll: float = 1000,
                                   bet_amount: float = 10) -> Dict[str, Any]:
        """
        分析策略性能
        
        Args:
            strategy: 要分析的策略
            shoe_ids: 要分析的鞋子ID列表，None为全部
            initial_bankroll: 初始资金
            bet_amount: 每次投注金额
            
        Returns:
            分析结果字典
        """
        self.logger.info(f"开始分析策略 {strategy.strategy_name} 的性能")
        
        if shoe_ids is None:
            shoe_ids = self._get_all_shoe_ids()
            
        results = {
            'strategy_name': strategy.strategy_name,
            'total_shoes': len(shoe_ids),
            'total_bets': 0,
            'total_wins': 0,
            'total_losses': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'consecutive_wins_max': 0,
            'consecutive_losses_max': 0,
            'bet_distribution': defaultdict(int),
            'performance_by_shoe': [],
            'monthly_performance': defaultdict(lambda: {'pnl': 0, 'bets': 0}),
            'risk_metrics': {}
        }
        
        all_pnls = []
        all_bet_history = []
        
        for shoe_id in shoe_ids:
            # 模拟每个鞋子
            shoe_result = self.simulator.simulate_strategy_on_shoe(
                strategy=strategy,
                shoe_id=shoe_id,
                initial_bankroll=initial_bankroll,
                fixed_bet_amount=bet_amount
            )
            
            if shoe_result['success']:
                bet_history = shoe_result['bet_history']
                final_bankroll = shoe_result['final_bankroll']
                shoe_pnl = final_bankroll - initial_bankroll
                
                # 累计统计
                results['total_bets'] += len(bet_history)
                results['total_pnl'] += shoe_pnl
                all_pnls.append(shoe_pnl)
                all_bet_history.extend(bet_history)
                
                # 记录每个鞋子的表现
                shoe_info = self._get_shoe_info(shoe_id)
                results['performance_by_shoe'].append({
                    'shoe_id': shoe_id,
                    'date': shoe_info.get('start_time', ''),
                    'bets': len(bet_history),
                    'pnl': shoe_pnl,
                    'final_bankroll': final_bankroll
                })
                
                # 按月统计
                if shoe_info.get('start_time'):
                    month_key = shoe_info['start_time'][:7]  # YYYY-MM
                    results['monthly_performance'][month_key]['pnl'] += shoe_pnl
                    results['monthly_performance'][month_key]['bets'] += len(bet_history)
        
        # 计算详细统计
        if all_bet_history:
            results.update(self._calculate_detailed_stats(all_bet_history, initial_bankroll))
            
        # 计算风险指标
        if all_pnls:
            results['risk_metrics'] = self._calculate_risk_metrics(all_pnls, all_bet_history, initial_bankroll)
            
        self.logger.info(f"策略 {strategy.strategy_name} 分析完成")
        return results
    
    def compare_strategies(self, strategies: List[BaseStrategy], 
                          shoe_ids: List[int] = None,
                          initial_bankroll: float = 1000,
                          bet_amount: float = 10) -> Dict[str, Any]:
        """
        比较多个策略的性能
        
        Args:
            strategies: 策略列表
            shoe_ids: 要分析的鞋子ID列表
            initial_bankroll: 初始资金
            bet_amount: 每次投注金额
            
        Returns:
            比较结果
        """
        self.logger.info(f"开始比较 {len(strategies)} 个策略")
        
        comparison_results = {
            'strategies': [],
            'ranking': [],
            'summary': {}
        }
        
        # 分析每个策略
        for strategy in strategies:
            result = self.analyze_strategy_performance(
                strategy, shoe_ids, initial_bankroll, bet_amount
            )
            comparison_results['strategies'].append(result)
            
        # 排名 - 按照夏普比率排序
        ranking = sorted(
            comparison_results['strategies'],
            key=lambda x: x.get('sharpe_ratio', 0),
            reverse=True
        )
        
        comparison_results['ranking'] = [
            {
                'rank': i + 1,
                'strategy_name': strategy['strategy_name'],
                'total_pnl': strategy['total_pnl'],
                'win_rate': strategy['win_rate'],
                'sharpe_ratio': strategy.get('sharpe_ratio', 0),
                'max_drawdown': strategy.get('max_drawdown', 0)
            }
            for i, strategy in enumerate(ranking)
        ]
        
        # 总结
        if ranking:
            best_strategy = ranking[0]
            comparison_results['summary'] = {
                'best_strategy': best_strategy['strategy_name'],
                'best_pnl': best_strategy['total_pnl'],
                'best_win_rate': max(s['win_rate'] for s in comparison_results['strategies']),
                'best_sharpe': max(s.get('sharpe_ratio', 0) for s in comparison_results['strategies']),
                'total_strategies': len(strategies)
            }
            
        return comparison_results
    
    def find_optimal_parameters(self, strategy_class, 
                               parameter_ranges: Dict[str, List],
                               shoe_ids: List[int] = None,
                               initial_bankroll: float = 1000) -> Dict[str, Any]:
        """
        寻找策略的最优参数
        
        Args:
            strategy_class: 策略类
            parameter_ranges: 参数范围，如 {'bet_amount': [5, 10, 20]}
            shoe_ids: 测试鞋子
            initial_bankroll: 初始资金
            
        Returns:
            最优参数结果
        """
        self.logger.info(f"开始寻找策略 {strategy_class.__name__} 的最优参数")
        
        if shoe_ids is None:
            shoe_ids = self._get_all_shoe_ids()[:10]  # 限制为前10个鞋子以提高速度
            
        best_result = None
        best_params = None
        all_results = []
        
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        for params in param_combinations:
            try:
                # 创建策略实例
                if hasattr(strategy_class, '__init__'):
                    # 检查构造函数参数
                    strategy = strategy_class()
                else:
                    strategy = strategy_class()
                
                # 应用参数（如果策略支持）
                for param_name, param_value in params.items():
                    if hasattr(strategy, param_name):
                        setattr(strategy, param_name, param_value)
                
                # 测试策略
                bet_amount = params.get('bet_amount', 10)
                result = self.analyze_strategy_performance(
                    strategy, shoe_ids, initial_bankroll, bet_amount
                )
                
                result['parameters'] = params
                all_results.append(result)
                
                # 判断是否为最佳结果（基于夏普比率）
                if best_result is None or result.get('sharpe_ratio', 0) > best_result.get('sharpe_ratio', 0):
                    best_result = result
                    best_params = params
                    
            except Exception as e:
                self.logger.error(f"测试参数 {params} 时出错: {e}")
                continue
        
        return {
            'best_parameters': best_params,
            'best_result': best_result,
            'all_results': all_results,
            'total_combinations_tested': len(all_results)
        }
    
    def analyze_market_conditions(self, shoe_ids: List[int] = None) -> Dict[str, Any]:
        """
        分析市场条件和模式
        
        Args:
            shoe_ids: 要分析的鞋子ID列表
            
        Returns:
            市场分析结果
        """
        self.logger.info("开始分析市场条件")
        
        if shoe_ids is None:
            shoe_ids = self._get_all_shoe_ids()
            
        analysis = {
            'total_shoes': len(shoe_ids),
            'total_rounds': 0,
            'result_distribution': Counter(),
            'streak_analysis': {
                'player_streaks': [],
                'banker_streaks': [],
                'tie_streaks': []
            },
            'patterns': {
                'alternating_sequences': 0,
                'long_streaks': 0,
                'choppy_games': 0
            },
            'statistics': {
                'avg_rounds_per_shoe': 0,
                'player_percentage': 0,
                'banker_percentage': 0,
                'tie_percentage': 0
            }
        }
        
        all_results = []
        
        for shoe_id in shoe_ids:
            rounds = self._get_shoe_rounds(shoe_id)
            if not rounds:
                continue
                
            analysis['total_rounds'] += len(rounds)
            
            # 统计结果分布
            for round_data in rounds:
                result = round_data['result']
                analysis['result_distribution'][result] += 1
                all_results.append(result)
                
            # 分析连胜模式
            streaks = self._analyze_streaks(rounds)
            for result_type, streak_list in streaks.items():
                analysis['streak_analysis'][f'{result_type.lower()}_streaks'].extend(streak_list)
                
            # 分析游戏模式
            patterns = self._analyze_game_patterns(rounds)
            for pattern_type, count in patterns.items():
                analysis['patterns'][pattern_type] += count
        
        # 计算统计数据
        if analysis['total_rounds'] > 0:
            analysis['statistics']['avg_rounds_per_shoe'] = analysis['total_rounds'] / len(shoe_ids)
            total = analysis['total_rounds']
            analysis['statistics']['player_percentage'] = (analysis['result_distribution']['Player'] / total) * 100
            analysis['statistics']['banker_percentage'] = (analysis['result_distribution']['Banker'] / total) * 100
            analysis['statistics']['tie_percentage'] = (analysis['result_distribution']['Tie'] / total) * 100
            
        return analysis
    
    def generate_strategy_report(self, strategy: BaseStrategy, 
                               shoe_ids: List[int] = None,
                               output_file: str = None) -> str:
        """
        生成策略分析报告
        
        Args:
            strategy: 要分析的策略
            shoe_ids: 分析的鞋子ID
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        analysis_result = self.analyze_strategy_performance(strategy, shoe_ids)
        
        report = f"""
策略分析报告
================

策略名称: {analysis_result['strategy_name']}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 基本统计
- 总鞋数: {analysis_result['total_shoes']}
- 总投注次数: {analysis_result['total_bets']}
- 胜率: {analysis_result['win_rate']:.2%}
- 总盈亏: ${analysis_result['total_pnl']:.2f}

## 风险指标
- 最大回撤: ${analysis_result.get('max_drawdown', 0):.2f}
- 夏普比率: {analysis_result.get('sharpe_ratio', 0):.3f}
- 盈亏比: {analysis_result.get('profit_factor', 0):.2f}

## 详细统计
- 最大连胜: {analysis_result.get('consecutive_wins_max', 0)}
- 最大连败: {analysis_result.get('consecutive_losses_max', 0)}
- 平均盈利: ${analysis_result.get('average_win', 0):.2f}
- 平均亏损: ${analysis_result.get('average_loss', 0):.2f}

## 按鞋子表现
"""
        
        for shoe_data in analysis_result['performance_by_shoe'][:10]:  # 显示前10个
            report += f"- 鞋子 {shoe_data['shoe_id']}: {shoe_data['bets']}注, 盈亏 ${shoe_data['pnl']:.2f}\n"
            
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"报告已保存到 {output_file}")
            except Exception as e:
                self.logger.error(f"保存报告失败: {e}")
                
        return report
    
    def _get_all_shoe_ids(self) -> List[int]:
        """获取所有鞋子ID"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT shoe_id FROM shoes ORDER BY shoe_id DESC")
            return [row['shoe_id'] for row in cursor.fetchall()]
        finally:
            conn.close()
            
    def _get_shoe_info(self, shoe_id: int) -> Dict[str, Any]:
        """获取鞋子信息"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM shoes WHERE shoe_id = ?", (shoe_id,))
            row = cursor.fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()
            
    def _get_shoe_rounds(self, shoe_id: int) -> List[Dict[str, Any]]:
        """获取鞋子的所有轮次"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM rounds 
                WHERE shoe_id = ? 
                ORDER BY round_number_in_shoe
            """, (shoe_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
            
    def _calculate_detailed_stats(self, bet_history: List, initial_bankroll: float) -> Dict[str, Any]:
        """计算详细统计数据"""
        stats = {}
        
        if not bet_history:
            return stats
            
        # 基本统计
        wins = [bet for bet in bet_history if bet[4] > 0]  # bet[4] is payout
        losses = [bet for bet in bet_history if bet[4] < 0]
        
        stats['total_wins'] = len(wins)
        stats['total_losses'] = len(losses)
        stats['win_rate'] = len(wins) / len(bet_history) if bet_history else 0
        
        # 平均盈亏
        if wins:
            stats['average_win'] = sum(bet[4] for bet in wins) / len(wins)
        if losses:
            stats['average_loss'] = sum(abs(bet[4]) for bet in losses) / len(losses)
            
        # 计算连胜连败
        streaks = self._calculate_win_loss_streaks(bet_history)
        stats['consecutive_wins_max'] = max(streaks['win_streaks']) if streaks['win_streaks'] else 0
        stats['consecutive_losses_max'] = max(streaks['loss_streaks']) if streaks['loss_streaks'] else 0
        
        # 盈亏比
        total_wins_amount = sum(bet[4] for bet in wins)
        total_losses_amount = sum(abs(bet[4]) for bet in losses)
        if total_losses_amount > 0:
            stats['profit_factor'] = total_wins_amount / total_losses_amount
            
        # 投注分布
        bet_distribution = Counter()
        for bet in bet_history:
            bet_distribution[bet[1]] += 1  # bet[1] is bet_on
        stats['bet_distribution'] = dict(bet_distribution)
        
        return stats
    
    def _calculate_risk_metrics(self, pnls: List[float], bet_history: List, initial_bankroll: float) -> Dict[str, Any]:
        """计算风险指标"""
        metrics = {}
        
        if not pnls or not bet_history:
            return metrics
            
        # 计算资金曲线
        equity_curve = [initial_bankroll]
        for bet in bet_history:
            equity_curve.append(equity_curve[-1] + bet[4])  # bet[4] is payout
            
        # 最大回撤
        running_max = [equity_curve[0]]
        for equity in equity_curve[1:]:
            running_max.append(max(running_max[-1], equity))
            
        drawdowns = [running_max[i] - equity_curve[i] for i in range(len(equity_curve))]
        metrics['max_drawdown'] = max(drawdowns) if drawdowns else 0
        
        # 夏普比率
        if len(pnls) > 1:
            avg_return = sum(pnls) / len(pnls)
            std_return = math.sqrt(sum((pnl - avg_return) ** 2 for pnl in pnls) / (len(pnls) - 1))
            metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
            
        # 胜率
        wins = sum(1 for bet in bet_history if bet[4] > 0)
        metrics['win_rate'] = wins / len(bet_history) if bet_history else 0
        
        return metrics
    
    def _calculate_win_loss_streaks(self, bet_history: List) -> Dict[str, List[int]]:
        """计算连胜连败"""
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for bet in bet_history:
            if bet[4] > 0:  # Win
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                current_win_streak += 1
            else:  # Loss
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                current_loss_streak += 1
                
        # 添加最后的连胜/连败
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
            
        return {'win_streaks': win_streaks, 'loss_streaks': loss_streaks}
    
    def _analyze_streaks(self, rounds: List[Dict]) -> Dict[str, List[int]]:
        """分析游戏结果的连胜模式"""
        streaks = {'Player': [], 'Banker': [], 'Tie': []}
        
        if not rounds:
            return streaks
            
        current_result = rounds[0]['result']
        current_streak = 1
        
        for round_data in rounds[1:]:
            if round_data['result'] == current_result:
                current_streak += 1
            else:
                streaks[current_result].append(current_streak)
                current_result = round_data['result']
                current_streak = 1
                
        # 添加最后的连胜
        streaks[current_result].append(current_streak)
        
        return streaks
    
    def _analyze_game_patterns(self, rounds: List[Dict]) -> Dict[str, int]:
        """分析游戏模式"""
        patterns = {
            'alternating_sequences': 0,
            'long_streaks': 0,
            'choppy_games': 0
        }
        
        if len(rounds) < 3:
            return patterns
            
        # 检测交替模式
        alternating_count = 0
        for i in range(len(rounds) - 1):
            if rounds[i]['result'] != rounds[i + 1]['result']:
                alternating_count += 1
            else:
                if alternating_count >= 3:
                    patterns['alternating_sequences'] += 1
                alternating_count = 0
                
        # 检测长连胜
        streak_count = 1
        for i in range(len(rounds) - 1):
            if rounds[i]['result'] == rounds[i + 1]['result']:
                streak_count += 1
            else:
                if streak_count >= 5:
                    patterns['long_streaks'] += 1
                streak_count = 1
                
        # 简单的choppy判断
        if alternating_count / len(rounds) > 0.6:
            patterns['choppy_games'] = 1
            
        return patterns
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """生成参数组合"""
        if not parameter_ranges:
            return [{}]
            
        combinations = []
        param_names = list(parameter_ranges.keys())
        
        def generate_combinations(current_combo, param_index):
            if param_index >= len(param_names):
                combinations.append(current_combo.copy())
                return
                
            param_name = param_names[param_index]
            for value in parameter_ranges[param_name]:
                current_combo[param_name] = value
                generate_combinations(current_combo, param_index + 1)
                
        generate_combinations({}, 0)
        return combinations


# 使用示例
if __name__ == '__main__':
    from .strategy_rules import FollowTheShoeStrategy, AlternateStrategy
    
    analyzer = StrategyAnalyzer()
    
    # 分析单个策略
    follow_strategy = FollowTheShoeStrategy()
    result = analyzer.analyze_strategy_performance(follow_strategy)
    print(f"策略分析结果: {result}")
    
    # 比较多个策略
    strategies = [FollowTheShoeStrategy(), AlternateStrategy()]
    comparison = analyzer.compare_strategies(strategies)
    print(f"策略比较结果: {comparison}")
    
    # 生成报告
    report = analyzer.generate_strategy_report(follow_strategy)
    print(report) 