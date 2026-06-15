"""
Casualty Actuarial Probability Project:

This project analyzes motor vehicle crash data in New York City and estimates the following:
- The probability that a crash has at least one casualty
- The expected number of casualties per crash
- Simulated portfolio-level casualty outcomes
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imports machine-learning tools from scikit-learn (usually written as sklearn)
# Builds traditional machine learning models and performs predictive data analysis

# Logistic Regression is used when you want to predict a yes/no outcome.
#   (Ex: Will crash have at least one casualty?)
#   Also commonly used for classification, especially binary classification
#   (Ex: gives probabilities such as 'This crash has a 0.42 probability)
# PoissonRegressor is used for when you want to predict a count (as the name Poisson suggests)
#   (Ex: How many casualties do we expect from this crash? Could be 0.18, 1.35, 2.10, etc.)
from sklearn.linear_model import LogisticRegression, PoissonRegressor

# The following are 'scorekeepers' that function to help judge how good the predictions were
from sklearn.metrics import (
    # brier_score_loss measures how accurate probability predictions are by comparing the predicted probability to the real result.
    #   A lower score is better (performance of the model). Ex: A brier score of 0 is perfect.
    #   Brier score is calculated by taking the average of (predicted_probability - actual_result)^2
    #   Note: actual result is always a 1 or 0 for this project
    brier_score_loss,

    # mean_absolute_error measures the average size of the model's mistake for count predictions.
    #   Ex: If actual casualties are 2 and predicted casualties is 1.4, the error is 0.6. 
    #   It will do this for every row and then take the averages of the errors.
    #   So if the mean absolute error is 0.35, then the model is off by about 0.35 casualties per crash on average.
    mean_absolute_error,

    # mean_poisson_deviance is a more specialized error measure for count models like Poisson Regression.
    #   Checks how well the predicted casualty counts match the actual casualty counts, using 
    #       assumptions that fit count data.
    #   Ex: How well do the model's expected casualty counts match the real casualty counts?
    #       Calculated as 2 * [actual * log(actual / predicted) - (actual - predicted)]
    #       Note: if actual is 0, then it equals 0 as actual * log(...) where actual is 0 will be 0
    mean_poisson_deviance,

    # roc_auc_score measures how well the probability model ranks risky crashes
    #   compared with less risky crashes.
    #   ROC stands for Receiver Operating Characteristic (a curve to show how well a
    #       classification model separates positives from negatives)
    #       Ex: Positive means at least one casualty, Negative means no casualty
    #   AUC stands for 'Area Under Curve'
    #   Ex: (Does this model usually give higher casualty probabilities to crashes that actually
    #   had casualties?)
    roc_auc_score,
)

# Spreadsheet Data Information to learn about as we go
ROW_LIMIT = 20000
PORTFOLIO_CRASH_COUNT = 5000
SIMULATIONS = 5000
RANDOM_SEED = 2026
OUTPUT_FOLDER = "casualty_probability_outputs"

DATA_URL = (
    "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"
    "?$limit=20000"
    "&$order=crash_date%20DESC"
)

BOROUGH_CENTERS = {
    "BROOKLYN": (40.6782, -73.9442),
    "QUEENS": (40.7282, -73.7949),
    "MANHATTAN": (40.7831, -73.9712),
    "BRONX": (40.8448, -73.8648),
    "STATEN ISLAND": (40.5795, -74.1502),
}

