# 黄金策略逐笔成交导出说明

目录：`/workspace/backtest_gold/trades/`

## 共用设定
- 成本：往返 $0.30/oz
- 账户：$10,000 起
- 仓位：按初始止损风险 1% 权益，名义杠杆上限 5×
- 入场：信号次根开盘（无前瞻）
- 过滤：仅当 prior 日线收盘 > SMA50（日线按 22:00 UTC 换日）

## 列含义
| 列 | 含义 |
|----|------|
| entry_time_utc / exit_time_utc | 开平仓时间（UTC，ISO8601） |
| side | long / short |
| entry_price / exit_price | 开平仓价格 |
| stop_init | 初始止损价 |
| size_oz | 成交盎司数 |
| pnl_usd | 该笔美元盈亏（已扣成本） |
| pnl_R | 盈亏相对初始风险的 R 倍数 |
| return_pct | 相对下单前权益的百分比 |
| bars_held | 持仓 K 线根数 |
| exit_reason | 出场原因（trail/stop/donch_exit/mid_exit/ma_exit/end 等） |
| equity_after | 该笔结束后权益 |

## 文件一览

| 文件 | 策略 | 数据源 | 行数（成交笔数） |
|------|------|--------|------------------|
| `trades_h1_donch_sma50_duka.csv` | H1 Donchian20 + SMA50，2×ATR 跟踪 + Donchian10 | Dukascopy H1 全样本 | 1176 |
| `trades_h1_donch_sma50_hist.csv` | 同上 | HistData M1→H1 | 1126 |
| `trades_h2_donch_sma50_duka.csv` | H2 Donchian20 + SMA50（同出场规则） | Dukascopy H1 重采样 H2 | 616 |
| `trades_h2_donch_sma50_hist.csv` | 同上 | HistData→H2 | 606 |
| `trades_h2_atr_channel_sma50_duka.csv` | H2 ATR 通道（SMA20±2ATR）突破 + SMA50 | Dukascopy H2 | 659 |
| `trades_h1_bb_sma50_duka.csv` | H1 布林带(20,2)上轨突破 + SMA50 | Dukascopy H1 | 1373 |

## 数据路径
- Dukascopy：`/workspace/backtest_gold/xauusd_h1_2010_2026.parquet`（约 2010-01 → 2026-10）
- HistData：`/workspace/backtest_gold/xauusd_h1_histdata_2010_2026.parquet`（M1 ASCII 重采样，时区按 America/New_York→UTC）

逻辑复用自 `backtest_cross_source.py`（与此前回测一致）。
