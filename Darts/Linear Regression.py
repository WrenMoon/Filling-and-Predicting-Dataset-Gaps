import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.utils.missing_values import fill_missing_values

OriginalDataSet =AirPassengersDataset().load()

values = OriginalDataSet.values()
# values = np.arange(50, step=0.5)
values[10:20] = np.nan
values[43:62] = np.nan

GapDataset = TimeSeries.from_values(values)

fill_missing_values(GapDataset).plot(label="filled data set")
(GapDataset).plot(label="original data set")

plt.legend()
plt.show()