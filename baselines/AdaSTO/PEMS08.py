import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import AdaST

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS08'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 12  # Length of input sequence (L)
OUTPUT_LEN = 12  # Length of output sequence (L_pred)
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings.get('NORM_EACH_CHANNEL', False) # Whether to normalize each channel of the data
RESCALE = regular_settings.get('RESCALE', True) # Whether to rescale the data
NULL_VAL = regular_settings.get('NULL_VAL', 0.0) # Null value in the data

# Model architecture and parameters
MODEL_ARCH = AdaST
MODEL_PARAM = {
    "L": INPUT_LEN,
    "L_pred": OUTPUT_LEN,
    "d_model": 256,
    "num_layers": 1,
    # "num_heads": 4,
    "N": regular_settings.get('NUM_NODES', 170),
    "input_dim": 3,
    "output_dim": 1,
    "num_hours": 288
}
NUM_EPOCHS = 100

# Training-specific parameters
LEARNING_RATE = 0.001
LAMBDA_ORTHO = 0.0
LAMBDA_SPARSE = 0.0
LAMBDA_ERRSUP = 0.0
ERRSUP_MODE = "mse"
USE_RECONSTRUCTOR = False
RECON_PRETRAIN_EPOCHS = 10
RECON_LR = 0.001
CHANNELS = [32, 64, 32]

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'AdaST Configuration for PEMS08'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": LEARNING_RATE,
    "weight_decay": 0.0,
}

# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}

CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

# Additional training parameters specific to AdaST
CFG.TRAIN.LAMBDA_ORTHO = LAMBDA_ORTHO
CFG.TRAIN.LAMBDA_SPARSE = LAMBDA_SPARSE
CFG.TRAIN.LAMBDA_ERRSUP = LAMBDA_ERRSUP
CFG.TRAIN.ERRSUP_MODE = ERRSUP_MODE
CFG.TRAIN.USE_RECONSTRUCTOR = USE_RECONSTRUCTOR
CFG.TRAIN.RECON_PRETRAIN_EPOCHS = RECON_PRETRAIN_EPOCHS
CFG.TRAIN.RECON_LR = RECON_LR
CFG.TRAIN.CHANNELS = CHANNELS

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation