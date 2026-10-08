# quant_miner - Project Structure

```text
.
├── bitget
│   ├── BOT_batch
│   │   ├── strategies_E1
│   │   │   └── rules_files
│   │   │       └── rules_batch_topV.py
│   │   └── backtesting_cr.py
│   ├── BOT_crypto
│   │   ├── alerts
│   │   │   ├── __init__.py
│   │   │   └── telegram_notifier.py
│   │   ├── api
│   │   │   ├── static
│   │   │   │   ├── bots
│   │   │   │   │   ├── 00
│   │   │   │   │   │   └── favicon.jpg
│   │   │   │   │   ├── 01
│   │   │   │   │   └── E1
│   │   │   │   │       └── favicon.jpg
│   │   │   │   ├── css
│   │   │   │   │   └── dashboard.css
│   │   │   │   ├── img
│   │   │   │   └── js
│   │   │   │       └── dashboard.js
│   │   │   ├── templates
│   │   │   │   ├── base.html
│   │   │   │   └── dashboard.html
│   │   │   ├── backend.py
│   │   │   ├── __init__.py
│   │   │   └── metrics.py
│   │   ├── bot_utils
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   └── timeframes.py
│   │   ├── config
│   │   │   ├── utils
│   │   │   │   ├── connect_pass.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── settings.py
│   │   │   ├── strategies_00.py
│   │   │   └── strategies_E1.py
│   │   ├── core
│   │   │   ├── demo_operative.py
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py
│   │   │   ├── production_operative.py
│   │   │   └── split_brain_checker.py
│   │   ├── execution
│   │   │   ├── brokers
│   │   │   │   ├── base_client.py
│   │   │   │   ├── bitget_client.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── order_manager.py
│   │   │   ├── position_tracker.py
│   │   │   └── trade_logger.py
│   │   ├── market_data
│   │   │   ├── data_utils.py
│   │   │   ├── __init__.py
│   │   │   └── websocket_manager.py
│   │   ├── persistence
│   │   │   ├── bot_files_00
│   │   │   │   ├── BOT_orchestator_00.log
│   │   │   │   ├── bot_state_00.json
│   │   │   │   ├── bot_trades_00.xlsx
│   │   │   │   └── launcher_output_00.log
│   │   │   └── bot_files_E1
│   │   │       ├── BOT_orchestator_E1.log
│   │   │       ├── bot_state_E1.json
│   │   │       ├── bot_trades_E1.xlsx
│   │   │       └── launcher_output_E1.log
│   │   ├── quality_control
│   │   │   ├── analyzer.py
│   │   │   └── __init__.py
│   │   ├── risk_control
│   │   │   ├── exposure_calculator.py
│   │   │   ├── __init__.py
│   │   │   └── risk_limiter.py
│   │   ├── state
│   │   │   ├── candle_tracker.py
│   │   │   ├── __init__.py
│   │   │   └── state_manager.py
│   │   ├── strategies
│   │   │   ├── __init__.py
│   │   │   ├── strategy_loader.py
│   │   │   ├── strategy_processor.py
│   │   │   └── strategy_registry.py
│   │   ├── validation
│   │   │   ├── candle_validator.py
│   │   │   ├── __init__.py
│   │   │   └── validation_module.py
│   │   └── main.py
│   ├── broker_client
│   │   ├── broker_api
│   │   │   ├── api_client.py
│   │   │   └── __init__.py
│   │   ├── broker_config.py
│   │   └── __init__.py
│   ├── data_crypto
│   │   ├── data_cr
│   │   │   ├── 01_raw
│   │   │   │   ├── ADAUSDT_12H.csv
│   │   │   │   ├── ADAUSDT_12H.parquet
│   │   │   │   ├── ADAUSDT_1D.csv
│   │   │   │   ├── ADAUSDT_1D.parquet
│   │   │   │   ├── ADAUSDT_1H.csv
│   │   │   │   ├── ADAUSDT_1H.parquet
│   │   │   │   ├── ADAUSDT_4H.csv
│   │   │   │   ├── ADAUSDT_4H.parquet
│   │   │   │   ├── ADAUSDT_5m.csv
│   │   │   │   ├── ADAUSDT_5m.parquet
│   │   │   │   ├── ADAUSDT_6H.csv
│   │   │   │   ├── ADAUSDT_6H.parquet
│   │   │   │   ├── AVAXUSDT_12H.csv
│   │   │   │   ├── AVAXUSDT_12H.parquet
│   │   │   │   ├── AVAXUSDT_1D.csv
│   │   │   │   ├── AVAXUSDT_1D.parquet
│   │   │   │   ├── AVAXUSDT_1H.csv
│   │   │   │   ├── AVAXUSDT_1H.parquet
│   │   │   │   ├── AVAXUSDT_4H.csv
│   │   │   │   ├── AVAXUSDT_4H.parquet
│   │   │   │   ├── AVAXUSDT_5m.csv
│   │   │   │   ├── AVAXUSDT_5m.parquet
│   │   │   │   ├── AVAXUSDT_6H.csv
│   │   │   │   ├── AVAXUSDT_6H.parquet
│   │   │   │   ├── BCHUSDT_12H.csv
│   │   │   │   ├── BCHUSDT_12H.parquet
│   │   │   │   ├── BCHUSDT_1D.csv
│   │   │   │   ├── BCHUSDT_1D.parquet
│   │   │   │   ├── BCHUSDT_1H.csv
│   │   │   │   ├── BCHUSDT_1H.parquet
│   │   │   │   ├── BCHUSDT_4H.csv
│   │   │   │   ├── BCHUSDT_4H.parquet
│   │   │   │   ├── BCHUSDT_5m.csv
│   │   │   │   ├── BCHUSDT_5m.parquet
│   │   │   │   ├── BCHUSDT_6H.csv
│   │   │   │   ├── BCHUSDT_6H.parquet
│   │   │   │   ├── BNBUSDT_12H.csv
│   │   │   │   ├── BNBUSDT_12H.parquet
│   │   │   │   ├── BNBUSDT_1D.csv
│   │   │   │   ├── BNBUSDT_1D.parquet
│   │   │   │   ├── BNBUSDT_1H.csv
│   │   │   │   ├── BNBUSDT_1H.parquet
│   │   │   │   ├── BNBUSDT_4H.csv
│   │   │   │   ├── BNBUSDT_4H.parquet
│   │   │   │   ├── BNBUSDT_5m.csv
│   │   │   │   ├── BNBUSDT_5m.parquet
│   │   │   │   ├── BNBUSDT_6H.csv
│   │   │   │   ├── BNBUSDT_6H.parquet
│   │   │   │   ├── BTCUSDT_12H.csv
│   │   │   │   ├── BTCUSDT_12H.parquet
│   │   │   │   ├── BTCUSDT_1D.csv
│   │   │   │   ├── BTCUSDT_1D.parquet
│   │   │   │   ├── BTCUSDT_1H.csv
│   │   │   │   ├── BTCUSDT_1H.parquet
│   │   │   │   ├── BTCUSDT_4H.csv
│   │   │   │   ├── BTCUSDT_4H.parquet
│   │   │   │   ├── BTCUSDT_5m.csv
│   │   │   │   ├── BTCUSDT_5m.parquet
│   │   │   │   ├── BTCUSDT_6H.csv
│   │   │   │   ├── BTCUSDT_6H.parquet
│   │   │   │   ├── DOGEUSDT_12H.csv
│   │   │   │   ├── DOGEUSDT_12H.parquet
│   │   │   │   ├── DOGEUSDT_1D.csv
│   │   │   │   ├── DOGEUSDT_1D.parquet
│   │   │   │   ├── DOGEUSDT_1H.csv
│   │   │   │   ├── DOGEUSDT_1H.parquet
│   │   │   │   ├── DOGEUSDT_4H.csv
│   │   │   │   ├── DOGEUSDT_4H.parquet
│   │   │   │   ├── DOGEUSDT_5m.csv
│   │   │   │   ├── DOGEUSDT_5m.parquet
│   │   │   │   ├── DOGEUSDT_6H.csv
│   │   │   │   ├── DOGEUSDT_6H.parquet
│   │   │   │   ├── ETHUSDT_12H.csv
│   │   │   │   ├── ETHUSDT_12H.parquet
│   │   │   │   ├── ETHUSDT_1D.csv
│   │   │   │   ├── ETHUSDT_1D.parquet
│   │   │   │   ├── ETHUSDT_1H.csv
│   │   │   │   ├── ETHUSDT_1H.parquet
│   │   │   │   ├── ETHUSDT_4H.csv
│   │   │   │   ├── ETHUSDT_4H.parquet
│   │   │   │   ├── ETHUSDT_5m.csv
│   │   │   │   ├── ETHUSDT_5m.parquet
│   │   │   │   ├── ETHUSDT_6H.csv
│   │   │   │   ├── ETHUSDT_6H.parquet
│   │   │   │   ├── LINKUSDT_12H.csv
│   │   │   │   ├── LINKUSDT_12H.parquet
│   │   │   │   ├── LINKUSDT_1D.csv
│   │   │   │   ├── LINKUSDT_1D.parquet
│   │   │   │   ├── LINKUSDT_1H.csv
│   │   │   │   ├── LINKUSDT_1H.parquet
│   │   │   │   ├── LINKUSDT_4H.csv
│   │   │   │   ├── LINKUSDT_4H.parquet
│   │   │   │   ├── LINKUSDT_5m.csv
│   │   │   │   ├── LINKUSDT_5m.parquet
│   │   │   │   ├── LINKUSDT_6H.csv
│   │   │   │   ├── LINKUSDT_6H.parquet
│   │   │   │   ├── NEARUSDT_12H.csv
│   │   │   │   ├── NEARUSDT_12H.parquet
│   │   │   │   ├── NEARUSDT_1D.csv
│   │   │   │   ├── NEARUSDT_1D.parquet
│   │   │   │   ├── NEARUSDT_1H.csv
│   │   │   │   ├── NEARUSDT_1H.parquet
│   │   │   │   ├── NEARUSDT_4H.csv
│   │   │   │   ├── NEARUSDT_4H.parquet
│   │   │   │   ├── NEARUSDT_5m.csv
│   │   │   │   ├── NEARUSDT_5m.parquet
│   │   │   │   ├── NEARUSDT_6H.csv
│   │   │   │   ├── NEARUSDT_6H.parquet
│   │   │   │   ├── SOLUSDT_12H.csv
│   │   │   │   ├── SOLUSDT_12H.parquet
│   │   │   │   ├── SOLUSDT_1D.csv
│   │   │   │   ├── SOLUSDT_1D.parquet
│   │   │   │   ├── SOLUSDT_1H.csv
│   │   │   │   ├── SOLUSDT_1H.parquet
│   │   │   │   ├── SOLUSDT_4H.csv
│   │   │   │   ├── SOLUSDT_4H.parquet
│   │   │   │   ├── SOLUSDT_5m.csv
│   │   │   │   ├── SOLUSDT_5m.parquet
│   │   │   │   ├── SOLUSDT_6H.csv
│   │   │   │   ├── SOLUSDT_6H.parquet
│   │   │   │   ├── UNIUSDT_12H.csv
│   │   │   │   ├── UNIUSDT_12H.parquet
│   │   │   │   ├── UNIUSDT_1D.csv
│   │   │   │   ├── UNIUSDT_1D.parquet
│   │   │   │   ├── UNIUSDT_1H.csv
│   │   │   │   ├── UNIUSDT_1H.parquet
│   │   │   │   ├── UNIUSDT_4H.csv
│   │   │   │   ├── UNIUSDT_4H.parquet
│   │   │   │   ├── UNIUSDT_5m.csv
│   │   │   │   ├── UNIUSDT_5m.parquet
│   │   │   │   ├── UNIUSDT_6H.csv
│   │   │   │   ├── UNIUSDT_6H.parquet
│   │   │   │   ├── XLMUSDT_12H.csv
│   │   │   │   ├── XLMUSDT_12H.parquet
│   │   │   │   ├── XLMUSDT_1D.csv
│   │   │   │   ├── XLMUSDT_1D.parquet
│   │   │   │   ├── XLMUSDT_1H.csv
│   │   │   │   ├── XLMUSDT_1H.parquet
│   │   │   │   ├── XLMUSDT_4H.csv
│   │   │   │   ├── XLMUSDT_4H.parquet
│   │   │   │   ├── XLMUSDT_5m.csv
│   │   │   │   ├── XLMUSDT_5m.parquet
│   │   │   │   ├── XLMUSDT_6H.csv
│   │   │   │   ├── XLMUSDT_6H.parquet
│   │   │   │   ├── XRPUSDT_12H.csv
│   │   │   │   ├── XRPUSDT_12H.parquet
│   │   │   │   ├── XRPUSDT_1D.csv
│   │   │   │   ├── XRPUSDT_1D.parquet
│   │   │   │   ├── XRPUSDT_1H.csv
│   │   │   │   ├── XRPUSDT_1H.parquet
│   │   │   │   ├── XRPUSDT_4H.csv
│   │   │   │   ├── XRPUSDT_4H.parquet
│   │   │   │   ├── XRPUSDT_5m.csv
│   │   │   │   ├── XRPUSDT_5m.parquet
│   │   │   │   ├── XRPUSDT_6H.csv
│   │   │   │   └── XRPUSDT_6H.parquet
│   │   │   └── 04_split
│   │   │       ├── IS
│   │   │       │   └── crypto_2019-01_2026-09_IS
│   │   │       │       ├── ADAUSDT_12H.parquet
│   │   │       │       ├── ADAUSDT_1D.parquet
│   │   │       │       ├── ADAUSDT_1H.parquet
│   │   │       │       ├── ADAUSDT_4H.parquet
│   │   │       │       ├── ADAUSDT_6H.parquet
│   │   │       │       ├── AVAXUSDT_12H.parquet
│   │   │       │       ├── AVAXUSDT_1D.parquet
│   │   │       │       ├── AVAXUSDT_1H.parquet
│   │   │       │       ├── AVAXUSDT_4H.parquet
│   │   │       │       ├── AVAXUSDT_6H.parquet
│   │   │       │       ├── BCHUSDT_12H.parquet
│   │   │       │       ├── BCHUSDT_1D.parquet
│   │   │       │       ├── BCHUSDT_1H.parquet
│   │   │       │       ├── BCHUSDT_4H.parquet
│   │   │       │       ├── BCHUSDT_6H.parquet
│   │   │       │       ├── BNBUSDT_12H.parquet
│   │   │       │       ├── BNBUSDT_1D.parquet
│   │   │       │       ├── BNBUSDT_1H.parquet
│   │   │       │       ├── BNBUSDT_4H.parquet
│   │   │       │       ├── BNBUSDT_6H.parquet
│   │   │       │       ├── BTCUSDT_12H.parquet
│   │   │       │       ├── BTCUSDT_1D.parquet
│   │   │       │       ├── BTCUSDT_1H.parquet
│   │   │       │       ├── BTCUSDT_4H.parquet
│   │   │       │       ├── BTCUSDT_6H.parquet
│   │   │       │       ├── DOGEUSDT_12H.parquet
│   │   │       │       ├── DOGEUSDT_1D.parquet
│   │   │       │       ├── DOGEUSDT_1H.parquet
│   │   │       │       ├── DOGEUSDT_4H.parquet
│   │   │       │       ├── DOGEUSDT_6H.parquet
│   │   │       │       ├── ETHUSDT_12H.parquet
│   │   │       │       ├── ETHUSDT_1D.parquet
│   │   │       │       ├── ETHUSDT_1H.parquet
│   │   │       │       ├── ETHUSDT_4H.parquet
│   │   │       │       ├── ETHUSDT_6H.parquet
│   │   │       │       ├── LINKUSDT_12H.parquet
│   │   │       │       ├── LINKUSDT_1D.parquet
│   │   │       │       ├── LINKUSDT_1H.parquet
│   │   │       │       ├── LINKUSDT_4H.parquet
│   │   │       │       ├── LINKUSDT_6H.parquet
│   │   │       │       ├── NEARUSDT_12H.parquet
│   │   │       │       ├── NEARUSDT_1D.parquet
│   │   │       │       ├── NEARUSDT_1H.parquet
│   │   │       │       ├── NEARUSDT_4H.parquet
│   │   │       │       ├── NEARUSDT_6H.parquet
│   │   │       │       ├── SOLUSDT_12H.parquet
│   │   │       │       ├── SOLUSDT_1D.parquet
│   │   │       │       ├── SOLUSDT_1H.parquet
│   │   │       │       ├── SOLUSDT_4H.parquet
│   │   │       │       ├── SOLUSDT_6H.parquet
│   │   │       │       ├── UNIUSDT_12H.parquet
│   │   │       │       ├── UNIUSDT_1D.parquet
│   │   │       │       ├── UNIUSDT_1H.parquet
│   │   │       │       ├── UNIUSDT_4H.parquet
│   │   │       │       ├── UNIUSDT_6H.parquet
│   │   │       │       ├── XLMUSDT_12H.parquet
│   │   │       │       ├── XLMUSDT_1D.parquet
│   │   │       │       ├── XLMUSDT_1H.parquet
│   │   │       │       ├── XLMUSDT_4H.parquet
│   │   │       │       ├── XLMUSDT_6H.parquet
│   │   │       │       ├── XRPUSDT_12H.parquet
│   │   │       │       ├── XRPUSDT_1D.parquet
│   │   │       │       ├── XRPUSDT_1H.parquet
│   │   │       │       ├── XRPUSDT_4H.parquet
│   │   │       │       └── XRPUSDT_6H.parquet
│   │   │       └── OOS
│   │   │           └── crypto_2026-09_2026-09_OOS
│   │   ├── steps
│   │   │   ├── integrity.py
│   │   │   ├── step0_symbol_selection.py
│   │   │   ├── step1_extraction.py
│   │   │   ├── step3_cleaning.py
│   │   │   ├── step5_highlow.py
│   │   │   └── step7_split.py
│   │   └── main_data.py
│   ├── develop
│   │   ├── bitget_tools
│   │   │   ├── biget_fees.py
│   │   │   ├── bitget_leverage.py
│   │   │   ├── bitget_position_mode.py
│   │   │   ├── bitget_sell.py
│   │   │   ├── fees_fills.py
│   │   │   └── taxes.py
│   │   ├── calibrations
│   │   │   └── calibration_BLOCK_stepM_is.py
│   │   ├── live_vs_backtest
│   │   │   ├── candle_source_check.py
│   │   │   ├── live_C.py
│   │   │   └── strategy_re_backtest.py
│   │   └── misce
│   │       └── DSR
│   │           ├── comparsion_mbias.py
│   │           └── dsr.py
│   └── signals
│       ├── condition_bank.py
│       ├── indicators_bank.py
│       ├── __init__.py
│       └── signal_builder.py
├── core
│   ├── backtesters
│   │   ├── build_cmd.txt
│   │   ├── __init__.py
│   │   ├── setup.py
│   │   ├── ZX_compute_BT_NPY.cpython-312-x86_64-linux-gnu.so
│   │   ├── ZX_compute_BT_NPY.pyx
│   │   └── ZX_compute_BT_YPY.pyx
│   ├── engines
│   │   ├── FF_test.py
│   │   └── wfo_WF.py
│   ├── indicators
│   │   └── indicators_pool.py
│   ├── pipeline
│   │   ├── backtest_runner.py
│   │   ├── correlation.py
│   │   ├── multiverse.py
│   │   ├── signal_cleaning.py
│   │   ├── stepM_is.py
│   │   ├── stepM_oos.py
│   │   └── wfo.py
│   ├── research
│   │   ├── screening
│   │   │   ├── screen_engine.py
│   │   │   ├── screen_kernels_cpu.py
│   │   │   ├── screen_kernels_gpu.py
│   │   │   └── screen_report.py
│   │   ├── stages
│   │   │   ├── combos.py
│   │   │   ├── grids.py
│   │   │   ├── __init__.py
│   │   │   └── screen.py
│   │   ├── artifacts.py
│   │   └── __init__.py
│   ├── rule_mining
│   │   ├── rule_generator.py
│   │   ├── rule_runner.py
│   │   └── rule_writter.py
│   ├── runs
│   │   ├── __init__.py
│   │   ├── run_deploy.py
│   │   └── run_portfolio.py
│   ├── setup
│   │   ├── config_backtest.py
│   │   ├── config_core.py
│   │   ├── config_paths.py
│   │   ├── config_pipeline.py
│   │   ├── config_research.py
│   │   ├── config_symbols.py
│   │   └── __init__.py
│   ├── symbols
│   │   ├── __init__.py
│   │   └── universe.py
│   └── utils
│       ├── batch_metrics.py
│       ├── ohlcv_utils.py
│       ├── paralelization.py
│       ├── plotting.py
│       └── reporting.py
├── darwinex
│   ├── BOT_batch
│   │   ├── strategies_DZ
│   │   │   └── rules_files
│   │   │       ├── rules_batch_top1.py
│   │   │       ├── rules_batch_top2.py
│   │   │       └── rules_batch_top3.py
│   │   ├── backtesting_fx.py
│   │   └── profiler.py
│   ├── BOT_forex
│   │   ├── darwinex
│   │   │   └── live
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── mt5_connection.py
│   │   │       ├── strategies_config.py
│   │   │       └── strategies.py
│   │   ├── logs
│   │   │   └── live.log
│   │   ├── mt5
│   │   │   ├── config
│   │   │   │   └── mt5.ini
│   │   │   ├── Dockerfile
│   │   │   └── entrypoint.sh
│   │   ├── .spyproject
│   │   │   └── config
│   │   │       ├── backups
│   │   │       │   ├── codestyle.ini.bak
│   │   │       │   ├── encoding.ini.bak
│   │   │       │   ├── vcs.ini.bak
│   │   │       │   └── workspace.ini.bak
│   │   │       ├── defaults
│   │   │       │   ├── defaults-codestyle-0.2.0.ini
│   │   │       │   ├── defaults-encoding-0.2.0.ini
│   │   │       │   ├── defaults-vcs-0.2.0.ini
│   │   │       │   └── defaults-workspace-0.2.0.ini
│   │   │       ├── codestyle.ini
│   │   │       ├── encoding.ini
│   │   │       ├── vcs.ini
│   │   │       └── workspace.ini
│   │   ├── docker-compose.yml
│   │   └── start.sh
│   ├── BOT_research
│   │   ├── artifacts
│   │   │   ├── screen_combos
│   │   │   │   └── screen_IS.json
│   │   │   └── screen_grids
│   │   │       ├── screen_IS_1H.json
│   │   │       └── screen_IS_4H.json
│   │   ├── precompute
│   │   │   ├── caches
│   │   │   │   ├── screen_IS_1H_NPY_null85_sa100_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_c7e812a4.pkl
│   │   │   │   ├── screen_IS_1H_NPY_null85_sa20_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_d7bd424d.pkl
│   │   │   │   ├── screen_IS_1H_NPY_null85_sa40_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_7ef86bd0.pkl
│   │   │   │   ├── screen_IS_4H_NPY_null85_sa100_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_7c4b887f.pkl
│   │   │   │   ├── screen_IS_4H_NPY_null85_sa20_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_adde0b7e.pkl
│   │   │   │   └── screen_IS_4H_NPY_null85_sa40_tp0.5-1.0-1.5_sl0.5-1.0-1.5_20sym_63ea39ad.pkl
│   │   │   └── caches.py
│   │   ├── sweep
│   │   │   ├── sweep
│   │   │   │   └── configs
│   │   │   │       ├── luck_max-0.1_top_i-4.json
│   │   │   │       ├── luck_max-0.1_top_i-5.json
│   │   │   │       ├── luck_max-0.1_top_i-6.json
│   │   │   │       ├── luck_max-0.1_top_i-7.json
│   │   │   │       ├── luck_max-0.2_top_i-2.json
│   │   │   │       ├── luck_max-0.2_top_i-3.json
│   │   │   │       └── luck_max-0.2_top_i-4.json
│   │   │   ├── sweep_02
│   │   │   ├── sweep_analyze.py
│   │   │   ├── sweep_backtest_fx.py
│   │   │   └── sweep_research_fx.py
│   │   └── research_fx.py
│   ├── broker_client
│   │   ├── broker_api
│   │   │   ├── __init__.py
│   │   │   └── mt5_client.py
│   │   ├── broker_config.py
│   │   ├── generate_mt5_ini.py
│   │   └── __init__.py
│   ├── data_forex
│   │   ├── data_fx
│   │   │   └── 04_split
│   │   │       ├── IS
│   │   │       │   └── fx_2017-01_2024-12_IS
│   │   │       │       ├── AUDCAD_1H.parquet
│   │   │       │       ├── AUDCAD_4H.parquet
│   │   │       │       ├── AUDJPY_1H.parquet
│   │   │       │       ├── AUDJPY_4H.parquet
│   │   │       │       ├── AUDUSD_1H.parquet
│   │   │       │       ├── AUDUSD_4H.parquet
│   │   │       │       ├── CADJPY_1H.parquet
│   │   │       │       ├── CADJPY_4H.parquet
│   │   │       │       ├── CHFJPY_1H.parquet
│   │   │       │       ├── CHFJPY_4H.parquet
│   │   │       │       ├── EURAUD_1H.parquet
│   │   │       │       ├── EURAUD_4H.parquet
│   │   │       │       ├── EURCAD_1H.parquet
│   │   │       │       ├── EURCAD_4H.parquet
│   │   │       │       ├── EURCHF_1H.parquet
│   │   │       │       ├── EURCHF_4H.parquet
│   │   │       │       ├── EURGBP_1H.parquet
│   │   │       │       ├── EURGBP_4H.parquet
│   │   │       │       ├── EURJPY_1H.parquet
│   │   │       │       ├── EURJPY_4H.parquet
│   │   │       │       ├── EURUSD_1H.parquet
│   │   │       │       ├── EURUSD_4H.parquet
│   │   │       │       ├── GBPCAD_1H.parquet
│   │   │       │       ├── GBPCAD_4H.parquet
│   │   │       │       ├── GBPCHF_1H.parquet
│   │   │       │       ├── GBPCHF_4H.parquet
│   │   │       │       ├── GBPJPY_1H.parquet
│   │   │       │       ├── GBPJPY_4H.parquet
│   │   │       │       ├── GBPUSD_1H.parquet
│   │   │       │       ├── GBPUSD_4H.parquet
│   │   │       │       ├── NZDJPY_1H.parquet
│   │   │       │       ├── NZDJPY_4H.parquet
│   │   │       │       ├── NZDUSD_1H.parquet
│   │   │       │       ├── NZDUSD_4H.parquet
│   │   │       │       ├── USDCAD_1H.parquet
│   │   │       │       ├── USDCAD_4H.parquet
│   │   │       │       ├── USDCHF_1H.parquet
│   │   │       │       ├── USDCHF_4H.parquet
│   │   │       │       ├── USDJPY_1H.parquet
│   │   │       │       └── USDJPY_4H.parquet
│   │   │       └── OOS
│   │   │           └── fx_2024-01_2026-09_OOS
│   │   │               ├── AUDCAD_1H.parquet
│   │   │               ├── AUDCAD_4H.parquet
│   │   │               ├── AUDJPY_1H.parquet
│   │   │               ├── AUDJPY_4H.parquet
│   │   │               ├── AUDUSD_1H.parquet
│   │   │               ├── AUDUSD_4H.parquet
│   │   │               ├── CADJPY_1H.parquet
│   │   │               ├── CADJPY_4H.parquet
│   │   │               ├── CHFJPY_1H.parquet
│   │   │               ├── CHFJPY_4H.parquet
│   │   │               ├── EURAUD_1H.parquet
│   │   │               ├── EURAUD_4H.parquet
│   │   │               ├── EURCAD_1H.parquet
│   │   │               ├── EURCAD_4H.parquet
│   │   │               ├── EURCHF_1H.parquet
│   │   │               ├── EURCHF_4H.parquet
│   │   │               ├── EURGBP_1H.parquet
│   │   │               ├── EURGBP_4H.parquet
│   │   │               ├── EURJPY_1H.parquet
│   │   │               ├── EURJPY_4H.parquet
│   │   │               ├── EURUSD_1H.parquet
│   │   │               ├── EURUSD_4H.parquet
│   │   │               ├── GBPCAD_1H.parquet
│   │   │               ├── GBPCAD_4H.parquet
│   │   │               ├── GBPCHF_1H.parquet
│   │   │               ├── GBPCHF_4H.parquet
│   │   │               ├── GBPJPY_1H.parquet
│   │   │               ├── GBPJPY_4H.parquet
│   │   │               ├── GBPUSD_1H.parquet
│   │   │               ├── GBPUSD_4H.parquet
│   │   │               ├── NZDJPY_1H.parquet
│   │   │               ├── NZDJPY_4H.parquet
│   │   │               ├── NZDUSD_1H.parquet
│   │   │               ├── NZDUSD_4H.parquet
│   │   │               ├── USDCAD_1H.parquet
│   │   │               ├── USDCAD_4H.parquet
│   │   │               ├── USDCHF_1H.parquet
│   │   │               ├── USDCHF_4H.parquet
│   │   │               ├── USDJPY_1H.parquet
│   │   │               └── USDJPY_4H.parquet
│   │   ├── steps
│   │   │   ├── __init_.py
│   │   │   ├── integrity.py
│   │   │   ├── step0_symbol_selection.py
│   │   │   ├── step1_extraction.py
│   │   │   ├── step3_cleaning.py
│   │   │   ├── step5_highlow.py
│   │   │   └── step7_split.py
│   │   ├── .DS_Store
│   │   ├── main_data_fx.py
│   │   └── mt5_source.py
│   ├── develop
│   │   ├── calibrations
│   │   │   └── calibration_BLOCK_stepM_is.py
│   │   ├── live_vs_backtest
│   │   │   ├── brief_trades
│   │   │   │   ├── trades_full_000771_4H_long_RSI7lt60_AND_ADX14gt30.csv
│   │   │   │   ├── trades_full_053208_4H_long_RSI14lt40_AND_ADX14gt25_AND_HISTVOL30gtSMA_HISTVOL20.csv
│   │   │   │   └── trades_full_061887_4H_long_RSI14lt50_AND_HISTVOL10gtSMA_HISTVOL40_AND_HISTVOL30gtSMA_HISTVOL20.csv
│   │   │   ├── live_C.py
│   │   │   ├── production_trades.csv
│   │   │   ├── production_trades.py
│   │   │   └── strategy_re_backtest.py
│   │   └── misce
│   │       ├── CPU
│   │       │   └── stepM_CPU.py
│   │       └── DSR
│   │           ├── comparsion_mbias.py
│   │           └── dsr.py
│   └── signals
│       ├── indicators_bank.py
│       ├── __init__.py
│       └── signal_builder.py
├── .spyproject
│   └── config
│       ├── backups
│       │   ├── codestyle.ini.bak
│       │   ├── encoding.ini.bak
│       │   ├── vcs.ini.bak
│       │   └── workspace.ini.bak
│       ├── defaults
│       │   ├── defaults-codestyle-0.2.0.ini
│       │   ├── defaults-encoding-0.2.0.ini
│       │   ├── defaults-vcs-0.2.0.ini
│       │   └── defaults-workspace-0.2.0.ini
│       ├── codestyle.ini
│       ├── encoding.ini
│       ├── vcs.ini
│       └── workspace.ini
├── .gitignore
└── PROJECT_STRUCTURE.md

109 directories, 528 files
```
