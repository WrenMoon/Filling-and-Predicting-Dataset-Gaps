import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, ExponentialSmoothing
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import SunspotsDataset

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

# Read data:
series_sunspot = SunspotsDataset().load().astype(np.float32)

series_sunspot.plot()
check_seasonality(series_sunspot, max_lag=240)

train_sp, val_sp = series_sunspot.split_after(pd.Timestamp("19401001"))

scaler_sunspot = Scaler()
train_sp_scaled = scaler_sunspot.fit_transform(train_sp)
val_sp_scaled = scaler_sunspot.transform(val_sp)
series_sp_scaled = scaler_sunspot.transform(series_sunspot)

"the 'monthly sun spots' dataset has {} data points".format(len(series_sunspot))


my_model_sp = TransformerModel(
    batch_size=32,
    input_chunk_length=125,
    output_chunk_length=36,
    n_epochs=20,
    model_name="sun_spots_transformer",
    nr_epochs_val_period=5,
    d_model=16,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    random_state=42,
    optimizer_kwargs={"lr": 1e-3},
    save_checkpoints=True,
    force_reset=True,
)


my_model_sp.fit(series=train_sp_scaled, val_series=val_sp_scaled, verbose=True)

def backtest(testing_model):
    # Compute the backtest predictions with the two models
    pred_series = testing_model.historical_forecasts(
        series=series_sp_scaled,
        start=pd.Timestamp("19401001"),
        forecast_horizon=36,
        stride=10,
        retrain=False,
        verbose=True,
    )

    pred_series_ets = ExponentialSmoothing().historical_forecasts(
        series=series_sp_scaled,
        start=pd.Timestamp("19401001"),
        forecast_horizon=36,
        stride=10,
        retrain=True,
        verbose=True,
    )
    val_sp_scaled.plot(label="actual")
    pred_series.plot(label="our Transformer")
    pred_series_ets.plot(label="ETS")
    plt.legend()
    print("Transformer MAPE:", mape(pred_series, val_sp_scaled))
    print("ETS MAPE:", mape(pred_series_ets, val_sp_scaled))


best_model_sp = TransformerModel.load_from_checkpoint(
    model_name="sun_spots_transformer", best=True
)
backtest(best_model_sp)

plt.legend()
plt.show()