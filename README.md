An Artificial Neural Network Model for filling gaps in chlorophyll data across time using surrounding data points. Code written, tests, compares and analyses different versions of the ANN along with other traditional gap filling methods.

## Data splitting and evaluation

The 3-point and 9-point training scripts use the date window recorded in
`Data/Dates.csv`. After missing-target and missing-neighbor filtering, the
available observations are sorted chronologically and divided 80/20:
the earlier 80% is used to fit the model and the later 20% is retained as a
held-out validation set.

That 20% validation set is used directly for model monitoring. Early stopping
and the learning-rate-reduction callbacks monitor `val_loss`, so the validation
set is also used for hyperparameter-style model selection and is not a final,
untouched test set.

The 2015-08-23 to 2016-06-28 examples in `WaveletAnalysis.py` and `temp.py`
are separate synthetic-gap demonstrations. They introduce an artificial gap
in the previously observed `Est5` values, then compare predictions against those
saved true values. Under the current `Data/Dates.csv` training window
(2018-10-01 to 2020-06-30), every 2015-16 example date is outside the training
and validation windows and is therefore withheld from model fitting. The
2015-16 figures or reports must not be described as samples from the 80/20 split.
