# EPFL Machine Learning Course CS-433
Machine Learning Course, Fall 2016 (a.k.a. PCML)
Kaicheng-Yu's version
Group project 1
Repository for the lecture notes, labs and projects - resources, code templates and solutions

# Use method
Run the implmentations.py with given data matrix.

The following function would produce the best result.
```python
logistic_regression_best(y, tx, lambda_, gamma, max_iters)
```

# Structure of codes
All source codes are under folder projects/project1/scripts

## Logistic regression
model.py
    Learning engine for logistic regression

# example usage:
```python
from model import LogisticRegression

# Training data reading as y, tx
# Testing data reading as test_data, test_ids
# ...
# Do some manipulations

# Creation of logistic regression model
model = LogisticRegression((tx,y), regularizer="Lasso", regularizer_p=0.1)

# Model training
weight = model.train(lr=0.1, decay=0.5, max_iters=2000, early_stop=400)
pred_label = predict_labels(weight, t_data)
create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                      '/submission/removed_outlier_{}.csv'.format(title))


# Cross validation, for lambdas (for penalized terms)
# If your machine support matplotlib.pyplot, you could set plot=True
best_weights, best_lambda, (err_tr, err_te) = \
   logistic.cross_validation(4, lambdas, 'regularizer_p',
                             lr=0.1, batch_size=32,
                             max_iters=6000, early_stop=1000,
                             plot=True)

# load test.csv as test_data, test_ids
y_pred = []
for w in weights:
    _y_pred = logistic.__call__(test_data, w)
    y_pred += _y_pred
y_pred = np.average(y_pred)
y_pred[np.where(y_pred <= 0.5)] = -1
y_pred[np.where(y_pred > 0.5)] = 1
create_csv_submission(test_ids, y_pred, PATH_TO_CSV_SUBMISSION)
```




