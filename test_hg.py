"""
HG铜期货回测测试
基于COMEX铜期货2019-2024真实走势特征
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_hg_data(years=5, seed=42):
    """创建COMEX铜期货模拟日线数据

    真实走势特征（美元/磅）：
    - 2019: 2.55-2.95 区间震荡
    - 2020: 疫情暴跌至2.10，V型反转至3.55
    - 2021: 碳中和 + 新能源大牛市 3.50→4.80→4.30
    - 2022: 5.00跌至3.20，美联储加息下跌
    - 2023: 3.50-4.20 震荡筑底
    - 2024: 新能源需求反弹 3.80→5.10→4.00
    """
    np.random.seed(seed)
    start_year = 2025 - years
    dates = pd.date_range(f'{start_year}-01-01', '2024-12-31', freq='B')
    n = len(dates)

    yearly_anchors = {
        2019: [(1, 2.60), (60, 2.90), (130, 2.70), (200, 2.60), (252, 2.80)],
        2020: [(1, 2.80), (30, 2.50), (55, 2.10), (80, 2.40), (160, 3.00), (252, 3.55)],
        2021: [(1, 3.55), (50, 4.00), (100, 4.50), (130, 4.80), (180, 4.30), (220, 4.40), (252, 4.40)],
        2022: [(1, 4.40), (30, 4.70), (55, 5.00), (100, 4.20), (160, 3.40), (200, 3.20), (252, 3.80)],
        2023: [(1, 3.80), (50, 4.10), (100, 3.60), (150, 3.80), (200, 3.70), (252, 3.85)],
        2024: [(1, 3.85), (50, 3.90), (90, 4.60), (110, 5.10), (160, 4.40), (200, 4.20), (252, 4.00)],
    }

    prices = np.zeros(n)
    year_start_idx = 0

    for year in range(start_year, 2025):
        year_dates = dates[dates.year == year]
        year_n = len(year_dates)
        if year_n == 0:
            continue

        anchors = yearly_anchors.get(year, [(1, 3.80), (252, 3.80)])
        anchor_days = [a[0] for a in anchors]
        anchor_prices = [a[1] for a in anchors]
        anchor_days_norm = [int(d * year_n / 252) for d in anchor_days]
        anchor_days_norm[-1] = year_n - 1

        year_prices = np.interp(range(year_n), anchor_days_norm, anchor_prices)
        # 铜的日波动率约1.5%
        daily_noise = np.cumsum(np.random.normal(0, 0.010, year_n)) * year_prices[0] * 0.4
        year_prices = year_prices + daily_noise

        prices[year_start_idx:year_start_idx + year_n] = year_prices
        year_start_idx += year_n

    prices = prices[:n]
    prices = np.clip(prices, 1.50, 6.00)

    opens = prices * (1 + np.random.normal(0, 0.005, n))
    daily_vol = np.abs(np.diff(prices, prepend=prices[0])) / prices
    intraday_range = np.maximum(daily_vol * 2, 0.006)

    highs = np.maximum(prices, opens) * (1 + np.random.uniform(0.001, intraday_range))
    lows = np.minimum(prices, opens) * (1 - np.random.uniform(0.001, intraday_range))

    base_volume = 60000
    vol_factor = 1 + daily_vol * 40
    volumes = (base_volume * vol_factor * np.random.uniform(0.5, 1.5, n)).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'open': np.round(opens, 4),
        'high': np.round(highs, 4),
        'low': np.round(lows, 4),
        'close': np.round(prices, 4),
        'volume': volumes
    })
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    return df


def backtest_hg(data, initial_capital=50000, ma_period=20, atr_multiplier=2.5,
                max_position=5, min_stop_pct=0.01):
    """HG铜期货回测

    HG合约参数：
    - 合约乘数：25000磅/手
    - 保证金比例：约6%
    - 手续费：约$2.5/手（忽略不计 vs 合约价值）
    """
    from data_processor import DataProcessor
    from signal_generator import SignalGenerator
    from risk_manager import RiskManager, PositionSide

    processor = DataProcessor()
    data_with_ma = processor.calculate_ma(data, period=ma_period)
    data_with_ma = processor.calculate_atr(data_with_ma, period=14)
    data_with_ma = processor.calculate_adx(data_with_ma, period=14)

    generator = SignalGenerator(ma_period=ma_period)
    signals_data = generator.generate_signals(data_with_ma)
    signals_data = generator.add_signal_filters(signals_data, min_body_ratio=0.3, min_volume_ratio=1.0)

    # HG合约参数
    contract_multiplier = 25000  # 25000磅/手
    margin_rate = 0.06           # 保证金6%
    commission_per_lot = 2.50    # 每手$2.5

    capital = initial_capital
    position = 0
    entry_price = 0
    stop_price = 0
    extreme_price = 0
    trades = []
    cooldown = 0
    cooldown_period = 3

    for i in range(len(signals_data)):
        row = signals_data.iloc[i]
        price = row['close']
        signal = row['signal']

        if position == 0:
            if cooldown > 0:
                cooldown -= 1
                continue

            current_adx = row.get('adx', 0)
            if pd.notna(current_adx) and current_adx < 20:
                continue

            direction = None
            if signal == 1:
                direction = PositionSide.LONG
                prev_extreme = signals_data.iloc[i-1]['low'] if i > 0 else row['low']
            elif signal == -1:
                direction = PositionSide.SHORT
                prev_extreme = signals_data.iloc[i-1]['high'] if i > 0 else row['high']

            if direction:
                rm = RiskManager()
                stop = rm.calculate_stop_loss(price, prev_extreme, direction)

                if stop.stop_distance_pct < min_stop_pct:
                    continue

                # 仓位计算：基于2%风险
                risk_per_lot = abs(price - stop.stop_price) * contract_multiplier
                if risk_per_lot <= 0:
                    continue
                max_risk = capital * 0.02
                lots = min(int(max_risk / risk_per_lot), max_position)
                lots = max(lots, 1)

                # 检查保证金是否够
                margin_needed = price * contract_multiplier * margin_rate * lots
                if margin_needed > capital * 0.8:
                    lots = max(1, int(capital * 0.8 / (price * contract_multiplier * margin_rate)))

                pos_sign = 1 if direction == PositionSide.LONG else -1
                position = pos_sign * lots
                entry_price = price
                stop_price = stop.stop_price
                extreme_price = price
                capital -= commission_per_lot * lots

                trades.append({
                    'date': row['date'],
                    'type': 'BUY' if pos_sign > 0 else 'SELL',
                    'price': price,
                    'size': lots,
                    'stop': stop_price,
                })

        else:
            current_atr = row.get('atr', 0)

            if position > 0:
                if price > extreme_price:
                    extreme_price = price
                if current_atr > 0:
                    trail = extreme_price - atr_multiplier * current_atr
                    stop_price = max(stop_price, trail)
                if price <= stop_price:
                    pnl = (price - entry_price) * abs(position) * contract_multiplier
                    capital += pnl - commission_per_lot * abs(position)
                    trades.append({'date': row['date'], 'type': 'CLOSE_LONG', 'price': price, 'pnl': pnl})
                    position = 0; entry_price = 0; stop_price = 0; extreme_price = 0
                    cooldown = cooldown_period

            elif position < 0:
                if price < extreme_price:
                    extreme_price = price
                if current_atr > 0:
                    trail = extreme_price + atr_multiplier * current_atr
                    stop_price = min(stop_price, trail)
                if price >= stop_price:
                    pnl = (entry_price - price) * abs(position) * contract_multiplier
                    capital += pnl - commission_per_lot * abs(position)
                    trades.append({'date': row['date'], 'type': 'CLOSE_SHORT', 'price': price, 'pnl': pnl})
                    position = 0; entry_price = 0; stop_price = 0; extreme_price = 0
                    cooldown = cooldown_period

    # 统计
    closed = [t for t in trades if 'pnl' in t]
    wins = [t for t in closed if t['pnl'] > 0]
    losses = [t for t in closed if t['pnl'] < 0]
    total = len(closed)
    win_rate = len(wins) / total if total > 0 else 0
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital,
        'total_trades': total,
        'win_rate': win_rate,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades,
    }


def main():
    print("=" * 60)
    print("   COMEX 铜期货 (HG) MA20 趋势跟踪策略回测")
    print("=" * 60)

    # --- 单次详细回测 ---
    print("\n【1】单次回测（seed=42, 5年）\n")
    data = create_hg_data(years=5, seed=42)
    print(f"数据量: {len(data)} 交易日, "
          f"价格区间: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    r = backtest_hg(data)
    print(f"\n初始资金:  ${r['initial_capital']:>12,.2f}")
    print(f"最终资金:  ${r['final_capital']:>12,.2f}")
    print(f"总收益率:   {r['total_return']*100:>+10.1f}%")
    print(f"交易次数:   {r['total_trades']:>10}")
    print(f"胜率:       {r['win_rate']*100:>10.1f}%")
    print(f"盈亏比:     {r['profit_factor']:>10.2f}")
    print(f"平均盈利:  ${r['avg_win']:>12,.2f}")
    print(f"平均亏损:  ${r['avg_loss']:>12,.2f}")

    print("\n交易明细:")
    for t in r['trades']:
        if 'pnl' in t:
            flag = "✓" if t['pnl'] > 0 else "✗"
            print(f"  {flag} {t['date'].strftime('%Y-%m-%d')} {t['type']:<12} "
                  f"@ ${t['price']:.4f}  P&L: ${t['pnl']:>+10,.2f}")
        else:
            print(f"  → {t['date'].strftime('%Y-%m-%d')} {t['type']:<12} "
                  f"@ ${t['price']:.4f}  stop=${t['stop']:.4f}  {t['size']}手")

    # --- 多种子稳定性测试 ---
    print("\n" + "=" * 60)
    print("【2】稳定性测试（8组随机种子）\n")
    print(f"  {'Seed':>6} | {'收益率':>9} | {'交易数':>6} | {'胜率':>6} | {'盈亏比':>6}")
    print("  " + "-" * 48)

    all_results = []
    for seed in [42, 123, 456, 789, 1024, 2025, 9999, 31415]:
        data = create_hg_data(years=5, seed=seed)
        r = backtest_hg(data)
        all_results.append(r)
        ret = r['total_return'] * 100
        print(f"  {seed:>6} | {ret:>+8.1f}% | {r['total_trades']:>5}  | "
              f"{r['win_rate']*100:>5.1f}% | {r['profit_factor']:>5.2f}")

    avg_ret = np.mean([r['total_return'] for r in all_results]) * 100
    avg_wr = np.mean([r['win_rate'] for r in all_results]) * 100
    wins = sum(1 for r in all_results if r['total_return'] > 0)
    print("  " + "-" * 48)
    print(f"  {'平均':>6} | {avg_ret:>+8.1f}% |        | {avg_wr:>5.1f}% |")
    print(f"  盈利率:  {wins}/{len(all_results)}")

    # --- 对比螺纹钢 ---
    print("\n" + "=" * 60)
    print("【3】HG vs RB 对比\n")
    from simple_backtest import create_test_data, simple_backtest
    rb_results = []
    hg_results = []
    for seed in [42, 123, 456, 789, 1024, 2025, 9999, 31415]:
        rb_data = create_test_data(years=5, seed=seed)
        rb_r = simple_backtest(rb_data)
        rb_results.append(rb_r['total_return'] * 100)

        hg_data = create_hg_data(years=5, seed=seed)
        hg_r = backtest_hg(hg_data)
        hg_results.append(hg_r['total_return'] * 100)

    print(f"  {'':>8} | {'RB螺纹钢':>10} | {'HG铜期货':>10}")
    print("  " + "-" * 36)
    print(f"  {'平均收益':>8} | {np.mean(rb_results):>+9.1f}% | {np.mean(hg_results):>+9.1f}%")
    print(f"  {'最好':>8} | {max(rb_results):>+9.1f}% | {max(hg_results):>+9.1f}%")
    print(f"  {'最差':>8} | {min(rb_results):>+9.1f}% | {min(hg_results):>+9.1f}%")
    print(f"  {'盈利率':>8} | {sum(1 for r in rb_results if r>0)}/8       | {sum(1 for r in hg_results if r>0)}/8")
    print("=" * 60)


if __name__ == "__main__":
    main()
