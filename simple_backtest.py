"""
MA20趋势跟踪策略 - 简化回测测试
验证策略逻辑而不使用Backtrader
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(years=5, seed=42):
    """创建基于真实螺纹钢统计特征的模拟日线数据

    模拟2019-2024年螺纹钢真实走势特征：
    - 2019: 3300-4100区间震荡，先涨后跌
    - 2020: 3100-3900，疫情暴跌后V型反转
    - 2021: 4000-6200，碳中和大牛市，10月暴跌
    - 2022: 3600-5200，震荡下行
    - 2023: 3400-3900，低位窄幅震荡
    - 2024: 3000-3900，持续下行

    Args:
        years: 模拟年数（从2024往前推）
        seed: 随机种子
    """
    np.random.seed(seed)
    start_year = 2025 - years
    dates = pd.date_range(f'{start_year}-01-01', '2024-12-31', freq='B')
    n = len(dates)

    # --- 基于真实行情构造价格路径 ---
    # 年度关键节点价格（模拟螺纹钢真实走势）
    yearly_anchors = {
        2019: [(1, 3400), (60, 3900), (130, 4100), (200, 3700), (244, 3500)],
        2020: [(1, 3500), (30, 3100), (60, 3400), (180, 3700), (244, 3900)],
        2021: [(1, 4200), (80, 5200), (140, 5000), (180, 6100), (200, 4600), (244, 4600)],
        2022: [(1, 4500), (50, 5100), (100, 4600), (150, 4000), (200, 3800), (244, 3900)],
        2023: [(1, 3900), (60, 3700), (120, 3600), (180, 3800), (244, 3600)],
        2024: [(1, 3800), (60, 3900), (120, 3600), (180, 3400), (244, 3100)],
    }

    prices = np.zeros(n)
    year_start_idx = 0

    for year in range(start_year, 2025):
        # 当年有多少个交易日
        year_dates = dates[(dates.year == year)]
        year_n = len(year_dates)
        if year_n == 0:
            continue

        anchors = yearly_anchors.get(year, [(1, 3800), (244, 3800)])
        # 插值构建价格骨架
        anchor_days = [a[0] for a in anchors]
        anchor_prices = [a[1] for a in anchors]
        # 归一化到实际交易日数
        anchor_days_norm = [int(d * year_n / 244) for d in anchor_days]
        anchor_days_norm[-1] = year_n - 1

        # 线性插值
        year_prices = np.interp(range(year_n), anchor_days_norm, anchor_prices)

        # 添加真实波动（日收益率标准差约1.2%）
        daily_noise = np.cumsum(np.random.normal(0, 0.008, year_n)) * year_prices[0] * 0.5
        year_prices = year_prices + daily_noise

        prices[year_start_idx:year_start_idx + year_n] = year_prices
        year_start_idx += year_n

    # 截断到实际长度
    prices = prices[:n]
    prices = np.clip(prices, 2500, 7000)

    # --- 生成OHLCV ---
    opens = prices * (1 + np.random.normal(0, 0.004, n))
    daily_vol = np.abs(np.diff(prices, prepend=prices[0])) / prices  # 基于价格变化的波动
    intraday_range = np.maximum(daily_vol * 2, 0.005)  # 至少0.5%日内波动

    highs = np.maximum(prices, opens) * (1 + np.random.uniform(0.001, intraday_range))
    lows = np.minimum(prices, opens) * (1 - np.random.uniform(0.001, intraday_range))

    # 成交量：波动大时放量
    base_volume = 150000
    vol_factor = 1 + daily_vol * 50  # 波动大则成交量大
    volumes = (base_volume * vol_factor * np.random.uniform(0.6, 1.4, n)).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'open': np.round(opens, 2),
        'high': np.round(highs, 2),
        'low': np.round(lows, 2),
        'close': np.round(prices, 2),
        'volume': volumes
    })

    # 确保价格逻辑正确
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    return df

def simple_backtest(data, initial_capital=100000, ma_period=20, commission=0.0003, slippage=0.001,
                    atr_stop_multiplier=2.0, max_position=10, min_stop_pct=0.01):
    """简化回测函数

    Args:
        atr_stop_multiplier: ATR止损倍数，默认2.0倍ATR跟踪止损
        max_position: 最大持仓手数，防止止损太近导致仓位过大
        min_stop_pct: 最小止损距离百分比，低于此值不开仓（说明波动太小）
    """
    logger.info("开始简化回测...")

    # 准备数据
    from data_processor import DataProcessor
    processor = DataProcessor()

    # 计算MA
    data_with_ma = processor.calculate_ma(data, period=ma_period)

    # 计算ATR用于跟踪止损
    data_with_ma = processor.calculate_atr(data_with_ma, period=14)

    # 计算ADX用于趋势过滤
    data_with_ma = processor.calculate_adx(data_with_ma, period=14)

    # 生成信号
    from signal_generator import SignalGenerator
    generator = SignalGenerator(ma_period=ma_period)
    signals_data = generator.generate_signals(data_with_ma)

    # 启用信号过滤器：过滤掉小实体和低量的虚假信号
    signals_data = generator.add_signal_filters(signals_data, min_body_ratio=0.3, min_volume_ratio=1.0)
    
    # 初始化回测状态
    capital = initial_capital
    position = 0  # 持仓数量
    entry_price = 0
    stop_price = 0
    extreme_price = 0  # 持仓期间极值（做多记最高，做空记最低）
    trades = []
    equity_curve = [initial_capital]
    cooldown = 0  # 信号冷却期计数器
    cooldown_period = 3  # 平仓后等3根K线再开仓
    
    # 回测逻辑
    for i in range(len(signals_data)):
        row = signals_data.iloc[i]
        current_price = row['close']
        signal = row['signal']
        
        # 无持仓时检查信号
        if position == 0:
            # 冷却期内不开仓
            if cooldown > 0:
                cooldown -= 1
                equity_curve.append(capital)
                continue

            # ADX趋势过滤：ADX < 20 表示震荡市，不开仓
            current_adx = row.get('adx', 0)
            if pd.notna(current_adx) and current_adx < 20:
                equity_curve.append(capital)
                continue

            if signal == 1:  # 做多信号
                # 计算止损
                from risk_manager import RiskManager, PositionSide
                risk_manager = RiskManager()
                
                # 使用前一根K线的极值
                prev_low = signals_data.iloc[i-1]['low'] if i > 0 else row['low']
                stop_result = risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    prev_extreme=prev_low,
                    direction=PositionSide.LONG
                )
                
                # 止损距离太小则跳过（窄幅震荡不值得做）
                if stop_result.stop_distance_pct < min_stop_pct:
                    equity_curve.append(capital)
                    continue

                # 计算仓位
                position_result = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    stop_price=stop_result.stop_price,
                    margin_rate=0.10,
                    contract_multiplier=10.0
                )

                # 开仓（限制最大仓位）
                position = min(position_result.position_size, max_position)
                entry_price = current_price
                stop_price = stop_result.stop_price
                extreme_price = current_price

                # 扣除手续费
                commission_cost = entry_price * position * 10 * commission
                capital -= commission_cost

                trades.append({
                    'date': row['date'],
                    'type': 'BUY',
                    'price': entry_price,
                    'size': position,
                    'stop_price': stop_price,
                    'capital': capital
                })

                logger.info(f"做多开仓: 价格={entry_price:.2f}, 数量={position}, 止损={stop_price:.2f}")

            elif signal == -1:  # 做空信号
                # 计算止损
                from risk_manager import RiskManager, PositionSide
                risk_manager = RiskManager()

                # 使用前一根K线的极值
                prev_high = signals_data.iloc[i-1]['high'] if i > 0 else row['high']
                stop_result = risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    prev_extreme=prev_high,
                    direction=PositionSide.SHORT
                )

                # 止损距离太小则跳过
                if stop_result.stop_distance_pct < min_stop_pct:
                    equity_curve.append(capital)
                    continue

                # 计算仓位
                position_result = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    stop_price=stop_result.stop_price,
                    margin_rate=0.10,
                    contract_multiplier=10.0
                )
                
                # 开仓（限制最大仓位）
                position = -min(position_result.position_size, max_position)
                entry_price = current_price
                stop_price = stop_result.stop_price
                extreme_price = current_price

                # 扣除手续费
                commission_cost = entry_price * abs(position) * 10 * commission
                capital -= commission_cost

                trades.append({
                    'date': row['date'],
                    'type': 'SELL',
                    'price': entry_price,
                    'size': position,
                    'stop_price': stop_price,
                    'capital': capital
                })

                logger.info(f"做空开仓: 价格={entry_price:.2f}, 数量={abs(position)}, 止损={stop_price:.2f}")
        
        # 有持仓时检查出场条件
        else:
            current_atr = row.get('atr', 0)

            if position > 0:  # 做多持仓
                # 更新极值和ATR跟踪止损
                if current_price > extreme_price:
                    extreme_price = current_price
                if current_atr > 0:
                    atr_trailing = extreme_price - atr_stop_multiplier * current_atr
                    stop_price = max(stop_price, atr_trailing)

                # 出场条件：触及止损 或 收盘跌破止损价
                should_exit = current_price <= stop_price

                if should_exit:
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * position * 10
                    capital += pnl

                    commission_cost = exit_price * abs(position) * 10 * commission
                    capital -= commission_cost

                    trades.append({
                        'date': row['date'],
                        'type': 'SELL',
                        'price': exit_price,
                        'size': position,
                        'pnl': pnl,
                        'capital': capital
                    })

                    logger.info(f"平多仓: 价格={exit_price:.2f}, 盈亏={pnl:.2f}, 止损={stop_price:.2f}")

                    position = 0
                    entry_price = 0
                    stop_price = 0
                    extreme_price = 0
                    cooldown = cooldown_period

            elif position < 0:  # 做空持仓
                # 更新极值和ATR跟踪止损
                if current_price < extreme_price:
                    extreme_price = current_price
                if current_atr > 0:
                    atr_trailing = extreme_price + atr_stop_multiplier * current_atr
                    stop_price = min(stop_price, atr_trailing)

                # 出场条件：触及止损 或 收盘突破止损价
                should_exit = current_price >= stop_price

                if should_exit:
                    exit_price = current_price
                    pnl = (entry_price - exit_price) * abs(position) * 10
                    capital += pnl

                    commission_cost = exit_price * abs(position) * 10 * commission
                    capital -= commission_cost

                    trades.append({
                        'date': row['date'],
                        'type': 'BUY',
                        'price': exit_price,
                        'size': position,
                        'pnl': pnl,
                        'capital': capital
                    })

                    logger.info(f"平空仓: 价格={exit_price:.2f}, 盈亏={pnl:.2f}, 止损={stop_price:.2f}")

                    position = 0
                    entry_price = 0
                    stop_price = 0
                    extreme_price = 0
                    cooldown = cooldown_period
        
        # 记录权益曲线
        equity_curve.append(capital)
    
    # 计算绩效指标
    total_return = (capital - initial_capital) / initial_capital
    winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
    losing_trades = len([t for t in trades if 'pnl' in t and t['pnl'] < 0])
    total_trades = winning_trades + losing_trades
    
    # 计算胜率
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 计算盈亏比
    if total_trades > 0:
        avg_win = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        avg_win = avg_loss = profit_factor = 0
    
    results = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades,
        'equity_curve': equity_curve
    }
    
    return results

def main():
    """主函数"""
    logger.info("开始MA20趋势跟踪策略简化回测测试...")
    
    # 创建测试数据
    test_data = create_test_data()
    logger.info(f"✓ 创建测试数据: {len(test_data)} 条记录")
    
    # 运行简化回测
    results = simple_backtest(test_data)
    
    # 打印结果
    print("\n" + "="*50)
    print("           简化回测结果")
    print("="*50)
    print(f"初始资金: {results['initial_capital']:,.2f} CNY")
    print(f"最终资金: {results['final_capital']:,.2f} CNY")
    print(f"总收益率: {results['total_return']*100:+.2f}%")
    print(f"总交易次数: {results['total_trades']}")
    print(f"盈利交易: {results['winning_trades']}")
    print(f"亏损交易: {results['losing_trades']}")
    print(f"胜率: {results['win_rate']*100:.2f}%")
    print(f"盈亏比: {results['profit_factor']:.2f}")
    print(f"平均盈利: {results['avg_win']:,.2f} CNY")
    print(f"平均亏损: {results['avg_loss']:,.2f} CNY")
    print("="*50)
    
    # 显示前几个交易
    if results['trades']:
        print(f"\n前5个交易:")
        for i, trade in enumerate(results['trades'][:5]):
            if 'pnl' in trade:
                print(f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} - {trade['type']} - "
                      f"价格: {trade['price']:.2f} - 盈亏: {trade['pnl']:,.2f}")
            else:
                print(f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} - {trade['type']} - "
                      f"价格: {trade['price']:.2f} - 开仓")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n✅ 简化回测测试完成!")
    print(f"策略在测试期间实现了 {results['total_return']*100:+.2f}% 的收益率")
    print(f"共进行了 {results['total_trades']} 笔交易，胜率 {results['win_rate']*100:.2f}%")