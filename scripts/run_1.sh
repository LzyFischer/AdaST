# ── Exp1 需要 AdaST + STAEformer ──
# python experiments/train.py -c baselines/AdaST/PurpleAir.py      -g 0  # ~2h
python -u experiments/train.py -c baselines/AdaST/PEMS08.py         -g 0 | tee experiments_rebuttal/ada8.txt # ~3h 
python -u experiments/train.py -c baselines/STAEformer/PurpleAir.py -g 0 | tee experiments_rebuttal/staep.txt # ~2h
python -u experiments/train.py -c baselines/STAEformer/PEMS08.py    -g 0 | tee experiments_rebuttal/stae8.txt # ~3h

# ── Exp2 不需要 checkpoint（quick mode 用合成数据）──

# ── Exp3 需要准备 Weather 数据集 ──
python -u scripts/data_preparation/Weather/generate_training_data.py
python -u experiments/train.py -c baselines/AdaST/Weather.py     | tee experiments_rebuttal/adaweather.txt -g 0
python -u experiments/train.py -c baselines/DLinear/Weather.py   | tee experiments_rebuttal/dlweather.txt -g 0
python -u experiments/train.py -c baselines/STAEformer/Weather.py | tee experiments_rebuttal/staeweather.txt -g 0  # 如果有这个config

# ── Exp4 需要 AdaST 在 PEMS07/PurpleAir 上的 checkpoint（同 Exp1）──