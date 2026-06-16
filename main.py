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
    #       classification model separates positives from negatives). Another way of describing
    #       this is that the ROC plots all the confusion matrices.
    #       Ex: Positive means at least one casualty, Negative means no casualty
    #   AUC stands for 'Area Under Curve'
    #   Ex: (Does this model usually give higher casualty probabilities to crashes that actually
    #   had casualties?)
    #   Takes the area under the curve of the Receiver Operating Characteristic.
    #   Use the ROC with the greater area, as this gives the highest performing probability
    #   model to, in this case, rank risky crashes compared with less risky crashes.
    roc_auc_score,
)

# Spreadsheet Data Information to learn about as we go
ROW_LIMIT = 20000
PORTFOLIO_CRASH_COUNT = 5000
SIMULATIONS = 5000
RANDOM_SEED = 2026

# Where we get our excel file and etc.
OUTPUT_FOLDER = "casualty_probability_outputs"

# The data we are working with
DATA_URL = (
    "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"

    # Gives up to 20000 rows
    "?$limit=20000"

    # Sorts the data by crash date and give it in descending order (DESC)
    "&$order=crash_date%20DESC"
)

# Shows where each center is roughly centered around
BOROUGH_CENTERS = {
    "BROOKLYN": (40.6782, -73.9442),
    "QUEENS": (40.7282, -73.7949),
    "MANHATTAN": (40.7831, -73.9712),
    "BRONX": (40.8448, -73.8648),
    "STATEN ISLAND": (40.5795, -74.1502),
}

# Demo Data used to demonstrate the actuarial
#   modeling pipeline in case the live data source is unavailable.
def make_demo_data(number_of_rows):
    random_numbers = np.random.default_rng(RANDOM_SEED)

    borough_choices = ["BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND"]
    
    factor_choices = [
        "Driver Inattention/Distraction",
        "Failure to Yield Right-of-Way",
        "Following Too Closely",
        "Unsafe Speed",
        "Traffic Control Disregarded",
        "Alcohol Involvement",
        "Unspecified",
    ]

    vehicle_choices = [
        "Sedan",
        "SUV",
        "Taxi",
        "Pick-up Truck",
        "Bus",
        "Motorcycle",
        "Bike",
    ]

    # Creates a sequence of 1000 dates and moves by 1 day at a time (freq="D")
    dates = pd.date_range("2022-01-01", periods = 1000, freq="D")

    # Randomly chooses 20000 crashes among the 1000 dates 
    #   (many crashes will be on the same date)
    crash_dates = random_numbers.choice(dates, size=number_of_rows)

    # Chooses random hour that each accident occurs
    hours = random_numbers.integers(0, 24, size=number_of_rows)

    # Chooses random borough that each accident occurs
    boroughs = random_numbers.choice(borough_choices, size=number_of_rows)

    # Random factor
    factors = random_numbers.choice(factor_choices, size=number_of_rows)

    # Random first vehicle to crash
    vehicles_1 = random_numbers.choice(vehicle_choices, size=number_of_rows)

    # Random second vehicle to crash (possibility that it is not with another car ([""])
    vehicles_2 = random_numbers.choice(vehicle_choices + [""], size=number_of_rows)

    # Creates an array 'FILLED' with the same value. So gives a 'number_of_rows' long 
    #   array with each value being 0.35
    injury_chance = np.full(number_of_rows, 0.35)

    # 'Where': If the factor is 'Unsafe Speed', add 0.20, otherwise add 0
    #   Works like an if/else across many rows at once
    #   Note: A | symbol means or, & means and
    injury_chance = injury_chance + np.where(factors == "Unsafe Speed", 0.20, 0)
    injury_chance = injury_chance + np.where(factors == "Alcohol Involvement", 0.25, 0)
    injury_chance = injury_chance + np.where(vehicles_1 == "Motorcycle", 0.25, 0)
    injury_chance = injury_chance + np.where(vehicles_1 == "Bike", 0.20, 0)
    injury_chance = injury_chance + np.where((hours >= 22) | (hours <= 4), 0.08, 0)
    
    # Modify each row so that if it is below 0.05, it will equal 0.05.
    #   If it is above 0.9, make it equal to 0.9
    injury_chance = np.clip(injury_chance, 0.05, 0.90)

    # Takes each injury_chance (1 trial) and uses a binomial random draw
    #   for yes/no outcomes, so it is either a 1 (casualty) or a 0 (no casualty)
    has_casualty = random_numbers.binomial(1, injury_chance)

    # If there is a casualty, then more than one person may have been hurt, so
    #   the casualty_count counts how many people were injured / passed away
    casualty_count = has_casualty * (1 + random_numbers.poisson(injury_chance))

    # A set chance 0.006 for how many people with a casualty actually end up dying
    killed = random_numbers.binomial(casualty_count, 0.006)
    injured = casualty_count - killed

    # Creates a table with each key becoming a column name.
    return pd.DataFrame(
        {
            "crash_date": crash_dates.astype(str),
            "crash_time": [str(hour) + ":00" for hour in hours],
            "borough": boroughs,
            "number_of_persons_injured": injured,
            "number_of_persons_killed": killed,
            "contributing_factor_vehicle_1": factors,
            "vehicle_type_code1": vehicles_1,
            "vehicle_type_code2": vehicles_2,
        }
    )

