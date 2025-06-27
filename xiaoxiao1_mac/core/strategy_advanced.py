"""
高级策略模块 - 包含更智能的百家乐策略
包括模式识别、动态投注、风险管理等高级功能
"""

from collections import deque, Counter
from typing import Dict, List, Any, Optional
import statistics
import random

from .strategy_rules import BaseStrategy
from .utils import get_db_connection


class PatternRecognitionStrategy(BaseStrategy):
    """基于模式识别的策略"""
    
    def __init__(self, pattern_length: int = 10):
        super().__init__(strategy_name="PatternRecognition")
        self.pattern_length = pattern_length
        
    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
            
        recent_results = self.get_last_n_rounds(shoe_id, n=self.pattern_length)
        
        if len(recent_results) < 3:
            return {'bet_on': 'Player' if 'Player' in available_bets else available_bets[0] if available_bets else None,
                   'reason': "Insufficient data for pattern analysis"}
        
        # 简单的模式：如果最近3个结果相同，则选择相反的
        if len(set(recent_results[-3:])) == 1:
            last_result = recent_results[-1]
            if last_result == 'Player' and 'Banker' in available_bets:
                return {'bet_on': 'Banker', 'reason': "Pattern: Breaking Player streak"}
            elif last_result == 'Banker' and 'Player' in available_bets:
                return {'bet_on': 'Player', 'reason': "Pattern: Breaking Banker streak"}
                
        return {'bet_on': available_bets[0] if available_bets else None, 'reason': "Pattern: Default choice"}


class DynamicBettingStrategy(BaseStrategy):
    """动态投注策略 - 根据表现调整投注"""
    
    def __init__(self, base_bet: float = 10, max_bet: float = 100, 
                 win_multiplier: float = 1.2, loss_multiplier: float = 0.8):
        super().__init__(strategy_name="DynamicBetting")
        self.base_bet = base_bet
        self.max_bet = max_bet
        self.win_multiplier = win_multiplier
        self.loss_multiplier = loss_multiplier
        self.current_bet_size = base_bet
        self.recent_performance = deque(maxlen=10)  # 最近10次的表现
        
    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
            
        # 更新投注大小基于最近表现
        self._update_bet_size()
        
        # 基本的跟牌策略
        last_rounds = self.get_last_n_rounds(shoe_id, n=1)
        
        if not last_rounds:
            bet_choice = 'Player' if 'Player' in available_bets else available_bets[0] if available_bets else None
        else:
            last_result = last_rounds[0]
            bet_choice = last_result if last_result in available_bets else available_bets[0] if available_bets else None
            
        return {
            'bet_on': bet_choice,
            'bet_amount': self.current_bet_size,
            'reason': f"Dynamic betting: ${self.current_bet_size:.2f} based on recent performance"
        }
    
    def _update_bet_size(self):
        """根据最近表现更新投注大小"""
        if not self.recent_performance:
            return
            
        # 计算最近胜率
        recent_wins = sum(1 for result in self.recent_performance if result > 0)
        win_rate = recent_wins / len(self.recent_performance)
        
        if win_rate > 0.6:  # 表现好，增加投注
            self.current_bet_size = min(self.current_bet_size * self.win_multiplier, self.max_bet)
        elif win_rate < 0.4:  # 表现差，减少投注
            self.current_bet_size = max(self.current_bet_size * self.loss_multiplier, self.base_bet * 0.5)
            
    def record_result(self, payout: float):
        """记录投注结果"""
        self.recent_performance.append(payout)


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, trend_length: int = 5):
        super().__init__(strategy_name="TrendFollowing")
        self.trend_length = trend_length
        
    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
            
        recent_results = self.get_last_n_rounds(shoe_id, n=self.trend_length)
        
        if len(recent_results) < 3:
            return {'bet_on': None, 'reason': "Insufficient data for trend analysis"}
        
        # 分析趋势
        result_counts = Counter(recent_results)
        most_common = result_counts.most_common(1)[0]
        
        if most_common[1] >= len(recent_results) * 0.6:  # 60%以上是同一结果
            if most_common[0] in available_bets:
                return {'bet_on': most_common[0], 'reason': f"Following {most_common[0]} trend"}
                
        return {'bet_on': None, 'reason': "No clear trend"}


class RiskManagedStrategy(BaseStrategy):
    """风险管理策略"""
    
    def __init__(self, max_consecutive_losses: int = 3, 
                 loss_reduction_factor: float = 0.5,
                 recovery_multiplier: float = 1.5):
        super().__init__(strategy_name="RiskManaged")
        self.max_consecutive_losses = max_consecutive_losses
        self.loss_reduction_factor = loss_reduction_factor
        self.recovery_multiplier = recovery_multiplier
        self.consecutive_losses = 0
        self.current_risk_level = 1.0  # 1.0 = normal, <1.0 = conservative, >1.0 = aggressive
        
    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
            
        # 检查最近的表现来调整风险水平
        self._assess_risk_level(shoe_id)
        
        # 如果连续亏损过多，暂停投注
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {
                'bet_on': None,
                'reason': f"Risk management: {self.consecutive_losses} consecutive losses, skipping round"
            }
            
        # 根据风险水平选择策略
        if self.current_risk_level < 0.7:  # 高风险，保守投注
            return self._conservative_bet(shoe_id, available_bets)
        elif self.current_risk_level > 1.3:  # 低风险，积极投注
            return self._aggressive_bet(shoe_id, available_bets)
        else:  # 正常风险
            return self._normal_bet(shoe_id, available_bets)
    
    def _assess_risk_level(self, shoe_id):
        """评估当前风险水平"""
        recent_results = self.get_last_n_rounds(shoe_id, n=20)
        
        if len(recent_results) < 5:
            self.current_risk_level = 1.0
            return
            
        # 简单的风险评估：基于结果的可预测性
        result_counts = Counter(recent_results[-10:])
        entropy = self._calculate_entropy(result_counts)
        
        # 熵越高，风险越高（结果越随机）
        # 熵越低，风险越低（结果越可预测）
        if entropy > 1.5:
            self.current_risk_level = 1.5  # 高熵，低风险
        elif entropy < 0.8:
            self.current_risk_level = 0.6  # 低熵，高风险
        else:
            self.current_risk_level = 1.0
            
    def _calculate_entropy(self, counts: Counter) -> float:
        """计算信息熵"""
        total = sum(counts.values())
        if total == 0:
            return 0
            
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * (p.bit_length() - 1)  # 简化的熵计算
                
        return entropy
    
    def _conservative_bet(self, shoe_id, available_bets):
        """保守投注策略"""
        # 选择历史胜率更高的选项
        if 'Banker' in available_bets:
            return {
                'bet_on': 'Banker',
                'risk_multiplier': 0.5,
                'reason': "Conservative: Banker has slight statistical edge"
            }
        return {'bet_on': available_bets[0] if available_bets else None, 'reason': "Conservative fallback"}
    
    def _normal_bet(self, shoe_id, available_bets):
        """正常投注策略"""
        last_results = self.get_last_n_rounds(shoe_id, n=3)
        if not last_results:
            return {'bet_on': 'Player' if 'Player' in available_bets else available_bets[0] if available_bets else None,
                   'reason': "Normal: Default choice"}
            
        # 简单的反向策略
        last_result = last_results[0]
        if last_result == 'Player' and 'Banker' in available_bets:
            return {'bet_on': 'Banker', 'reason': "Normal: Alternating strategy"}
        elif last_result == 'Banker' and 'Player' in available_bets:
            return {'bet_on': 'Player', 'reason': "Normal: Alternating strategy"}
        else:
            return {'bet_on': available_bets[0] if available_bets else None, 'reason': "Normal: Fallback"}
    
    def _aggressive_bet(self, shoe_id, available_bets):
        """积极投注策略"""
        # 寻找强趋势
        recent_results = self.get_last_n_rounds(shoe_id, n=5)
        if len(recent_results) >= 3:
            result_counts = Counter(recent_results[-3:])
            most_common = result_counts.most_common(1)[0]
            if most_common[1] >= 2 and most_common[0] in available_bets:
                return {
                    'bet_on': most_common[0],
                    'risk_multiplier': 1.5,
                    'reason': f"Aggressive: Following {most_common[0]} trend"
                }
                
        return {'bet_on': available_bets[0] if available_bets else None, 'reason': "Aggressive: No clear trend"}
    
    def record_loss(self):
        """记录一次亏损"""
        self.consecutive_losses += 1
        
    def record_win(self):
        """记录一次盈利"""
        self.consecutive_losses = 0


class AdaptiveMLStrategy(BaseStrategy):
    """自适应机器学习策略"""
    
    def __init__(self, adaptation_period: int = 50):
        super().__init__(strategy_name="AdaptiveML")
        self.adaptation_period = adaptation_period
        self.feature_weights = {
            'last_result': 0.3,
            'streak_length': 0.2,
            'alternating_pattern': 0.2,
            'result_distribution': 0.3
        }
        self.performance_history = deque(maxlen=adaptation_period)
        
    def decide_bet(self, shoe_id, current_round_number_in_shoe, available_bets=None):
        if available_bets is None:
            available_bets = ['Player', 'Banker']
            
        # 提取特征
        features = self._extract_features(shoe_id)
        
        # 基于特征计算概率
        probabilities = self._calculate_probabilities(features, available_bets)
        
        # 选择最高概率的选项
        if probabilities:
            best_choice = max(probabilities, key=probabilities.get)
            confidence = probabilities[best_choice]
            
            if confidence > 0.55:  # 只有当置信度足够高时才投注
                return {
                    'bet_on': best_choice,
                    'confidence': confidence,
                    'reason': f"ML prediction: {best_choice} (confidence: {confidence:.2%})"
                }
                
        return {'bet_on': None, 'reason': "ML: Insufficient confidence for bet"}
    
    def _extract_features(self, shoe_id) -> Dict[str, float]:
        """提取决策特征"""
        recent_results = self.get_last_n_rounds(shoe_id, n=20)
        
        features = {
            'last_result_player': 0,
            'last_result_banker': 0,
            'streak_length': 0,
            'alternating_score': 0,
            'player_ratio': 0,
            'banker_ratio': 0
        }
        
        if not recent_results:
            return features
            
        # 最后结果
        last_result = recent_results[0] if recent_results else None
        if last_result == 'Player':
            features['last_result_player'] = 1
        elif last_result == 'Banker':
            features['last_result_banker'] = 1
            
        # 连胜长度
        streak_length = 1
        for i in range(1, len(recent_results)):
            if recent_results[i] == last_result:
                streak_length += 1
            else:
                break
        features['streak_length'] = min(streak_length, 10) / 10  # 归一化
        
        # 交替模式评分
        alternating_count = 0
        for i in range(len(recent_results) - 1):
            if recent_results[i] != recent_results[i + 1]:
                alternating_count += 1
        features['alternating_score'] = alternating_count / max(len(recent_results) - 1, 1)
        
        # 结果分布
        result_counts = Counter(recent_results)
        total = len(recent_results)
        features['player_ratio'] = result_counts.get('Player', 0) / total
        features['banker_ratio'] = result_counts.get('Banker', 0) / total
        
        return features
    
    def _calculate_probabilities(self, features: Dict[str, float], available_bets: List[str]) -> Dict[str, float]:
        """基于特征计算各选项的概率"""
        probabilities = {}
        
        for bet_option in available_bets:
            if bet_option == 'Player':
                prob = (
                    features['last_result_banker'] * self.feature_weights['last_result'] +  # 反向逻辑
                    (1 - features['streak_length']) * self.feature_weights['streak_length'] +  # 反趋势
                    features['alternating_score'] * self.feature_weights['alternating_pattern'] +
                    (1 - features['banker_ratio']) * self.feature_weights['result_distribution']
                )
            elif bet_option == 'Banker':
                prob = (
                    features['last_result_player'] * self.feature_weights['last_result'] +
                    (1 - features['streak_length']) * self.feature_weights['streak_length'] +
                    features['alternating_score'] * self.feature_weights['alternating_pattern'] +
                    (1 - features['player_ratio']) * self.feature_weights['result_distribution']
                )
            else:
                prob = 0.1  # Tie的基础概率
                
            probabilities[bet_option] = min(max(prob, 0.1), 0.9)  # 限制在合理范围内
            
        # 归一化概率
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
            
        return probabilities
    
    def adapt_weights(self, bet_result: str, actual_result: str):
        """根据结果调整特征权重"""
        was_correct = (bet_result == actual_result)
        self.performance_history.append(was_correct)
        
        # 如果表现不佳，随机调整权重
        if len(self.performance_history) >= 10:
            recent_accuracy = sum(self.performance_history[-10:]) / 10
            if recent_accuracy < 0.45:  # 表现差于随机
                # 小幅随机调整权重
                for feature in self.feature_weights:
                    adjustment = random.uniform(-0.05, 0.05)
                    self.feature_weights[feature] = max(0.1, min(0.5, 
                        self.feature_weights[feature] + adjustment))
                    
                # 重新归一化权重
                total_weight = sum(self.feature_weights.values())
                self.feature_weights = {k: v / total_weight for k, v in self.feature_weights.items()}


# 策略工厂函数
def create_advanced_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """创建高级策略实例"""
    strategy_map = {
        'pattern': PatternRecognitionStrategy,
        'dynamic': DynamicBettingStrategy,
        'trend': TrendFollowingStrategy,
        'risk': RiskManagedStrategy,
        'adaptive': AdaptiveMLStrategy
    }
    
    strategy_class = strategy_map.get(strategy_type.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
        
    return strategy_class(**kwargs)


# 使用示例
if __name__ == '__main__':
    # 测试模式识别策略
    pattern_strategy = PatternRecognitionStrategy(pattern_length=5)
    print(f"Created strategy: {pattern_strategy.strategy_name}")
    
    # 测试动态投注策略
    dynamic_strategy = DynamicBettingStrategy(base_bet=10, max_bet=50)
    print(f"Created strategy: {dynamic_strategy.strategy_name}")
    
    # 使用工厂函数
    trend_strategy = create_advanced_strategy('trend', trend_length=7)
    print(f"Created strategy: {trend_strategy.strategy_name}") 