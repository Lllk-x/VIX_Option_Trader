@echo off
cd /d "C:\Users\Qiao_0529\VIX_Options"
python vix_options_strategy.py --mode collect --snapshot_dir vix_snapshots >> logs\collect.log 2>&1
