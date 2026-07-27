
python experiments/train.py -c baselines/DLinear/PEMS08.py
python experiments/train.py -c baselines/DLinear/PurpleAir.py

python experiments/train.py -c baselines/PatchTST/PEMS08.py
python experiments/train.py -c baselines/PatchTST/PurpleAir.py

python experiments/train.py -c baselines/HimNet/ExchangeRate.py
python experiments/train.py -c baselines/HimNet/ETTh1.py
python experiments/train.py -c baselines/HimNet/METR-LA.py








"STID ExchangeRate"
    "STID ETTh1"
    "STID METR-LA"
    "DLinear ExchangeRate"
    "DLinear ETTh1"
    "DLinear METR-LA"
    "AdaST ETTh1"
        "AdaST ExchangeRate"
    "AdaST METR-LA"