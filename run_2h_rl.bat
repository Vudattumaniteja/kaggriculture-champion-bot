@echo off
cd /d "%~dp0"
echo =======================================================
echo LAUNCHING 2-HOUR RL SESSION (14 WORKERS + PRETRAINED)
echo =======================================================
python src/training/overnight_rl_pipeline.py --hours 2.0 --workers 14 --warm_start "grilling model/data/pretrained_champion.pt" --champion_weights "weights/alphagoat_world_champion.pt" --checkpoint_dir "weights/overnight_checkpoints" --log_path "data/overnight_training_log.json" --matches_per_iter 28 --batch_size 64 --epochs_per_iter 4 --lr 0.0002 --checkpoint_mins 15.0 --iterations 500
pause
