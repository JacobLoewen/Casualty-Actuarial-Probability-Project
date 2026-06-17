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

# Method that helps clean missing or messy borough data
#   Helps the project figure out the borough when the original dataset has missing borough values

def get_borough_from_zip(zip_value):
    zip_text = str(zip_value).strip()

    # "NAN" means pandas/NumPy may ahve marked it as missing data, so we just return ""
    if zip_text == "" or zip_text.upper() == "NAN":
        return ""
    
    # Takes the part of the zip_text that is before . and - in case of 10001.0 or 10001-1234
    zip_text = zip_text.split(".")[0].split("-")[0]

    try:
        zip_code = int(zip_text)
    except ValueError:
        return ""
    
    if 10000 <= zip_code <= 10292:
        return "MANHATTAN"
    if 10300 <= zip_code <= 10314:
        return "STATEN ISLAND"
    if 10400 <= zip_code <= 10475:
        return "BRONX"
    if 11200 <= zip_code <= 11256:
        return "BROOKLYN"
    if zip_code in [11004, 11005] or 11100 <= zip_code <= 11109 or 11300 <= zip_code <= 11697:
        return "QUEENS"

    return ""
    
# Able to also get the borough from coordinates if needed:
def get_borough_from_coordinates(latitude_value, longitude_value):
    try:
        latitude = float(latitude_value)
        longitude = float(longitude_value)
    except ValueError:
        # Could not get borough from coordinates
        return ""
    
    # Checks if coords are outside the broad New York City area
    if not (40.45 <= latitude <= 40.95 and -74.30 <= longitude <= -73.65):
        return ""
    
    # Comparing below, so start closest_distance at max
    closest_borough = ""
    closest_distance = float("inf")

    # Finds closest borough to where the coordinates are from
    for borough, center in BOROUGH_CENTERS.items():
        center_latitude, center_longitude = center
        distance = (latitude - center_latitude) ** 2 + (longitude - center_longitude) ** 2

        if distance < closest_distance:
            closest_borough = borough
            closest_distance = distance
        
    return closest_borough

# The final borough-cleaning function that uses the functions above.
    # Use borough if it exists. If not, then ZIP code. If not that, then coords.
    # If all else fails, label "UNKNOWN"

def get_clean_borough(row):
    borough_text = str(row.get("borough", "")).strip().upper()

    # If it already has a usable borough, return it.
    if borough_text not in ["", "NAN", "UNKNOWN"]:
        return borough_text
    
    zip_borough = get_borough_from_zip(row.get("zip_code", ""))
    if zip_borough != "":
        return zip_borough
    
    coordinate_borough = get_borough_from_coordinates(
        row.get("latitude", ""),
        row.get("longitude", ""),
    )
    if coordinate_borough != "":
        return coordinate_borough
    
    return "UNKNOWN"

# Time and category cleanup functions
def get_hour(time_value):
    time_text = str(time_value)
    pieces = time_text.split(":")

    try:
        return int(pieces[0])
    except ValueError:
        return -1
    
def get_hour_group(hour):
    if hour < 0:
        return "Unknown"
    if hour >= 5 and hour <= 9:
        return "Morning"
    if hour >= 10 and hour <= 14:
        return "Midday"
    if hour >= 15 and hour <= 22:
        return "Evening"
    return "Late night"

def get_factor_group(factor):
    factor_text = str(factor).upper()

    if "UNSAFE SPEED" in factor_text:
        return "Unsafe speed"
    if "ALCOHOL" in factor_text:
        return "Alcohol or drugs"
    if "FAILURE TO YIELD" in factor_text:
        return "Failure to yield"
    if "FOLLOWING TOO CLOSELY" in factor_text:
        return "Following too closely"
    if "TRAFFIC CONTROL" in factor_text:
        return "Traffic control"
    if "INATTENTION" in factor_text or "DISTRACTION" in factor_text:
        return "Driver distraction"
    if factor_text == "UNSPECIFIED" or factor_text == "NAN":
        return "Unspecified"
    return "Other"

def get_vehicle_group(vehicle):
    vehicle_text = str(vehicle).upper()

    if "BIKE" in vehicle_text or "BICYCLE" in vehicle_text:
        return "Bicycle"
    if "MOTORCYCLE" in vehicle_text:
        return "Motorcycle"
    if "BUS" in vehicle_text:
        return "Bus"
    if "TAXI" in vehicle_text:
        return "Taxi"
    if "TRUCK" in vehicle_text:
        return "Truck"
    if "SUV" in vehicle_text or "SPORT UTILITY" in vehicle_text:
        return "SUV"
    if "SEDAN" in vehicle_text:
        return "Passenger car"
    if vehicle_text == "UNKNOWN" or vehicle_text == "" or vehicle_text == "NAN":
        return "Unknown"
    return "Other"

# Creates the output folder where my CSV, Excel, and chart files will be saved
#   exist_ok=True makes it so that if the folder already exists, it doesn't crash
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Let the user know the casualty data is loading
print("Loading casualty data...")

try:
    # Attempts to read the actual data
    data = pd.read_csv(DATA_URL)
    data_source = "NYC Open Data motor vehicle collisions"
except Exception as error:
    # Reads demo data instead
    print("Could not download data, so demo data will be used.")
    print("Reason:", error)
    data = make_demo_data(ROW_LIMIT)
    data_source = "demo data"

print("Cleaning data...")

# Converts bad dates into missing values to prevent program from crashing
data["crash_date"] = pd.to_datetime(data["crash_date"], errors="coerce")

# apply is a pandas function that runs a function for every row
# dt stands for date/time. It is a pandas function. Allows you to get month, day, etc.
#   from date data.
data["hour"] = data["crash_time"].apply(get_hour)

# dropna removes rows that are n/a (has missing values, in this case,
#   the crash_date is specifically missing)
data = data.dropna(subset=["crash_date"])
data = data[data["hour"] >= 0]

data["month"] = data["crash_date"].dt.month
data["weekday"] = data["crash_date"].dt.day_name()
data["is_weekend"] = data["weekday"].isin(["Saturday", "Sunday"]).astype(int)
data["is_night"] = ((data["hour"] >= 22) | (data["hour"] <= 4)).astype(int)
data["hour_group"] = data["hour"].apply(get_hour_group)

# Cleaning borough, crash factor, vehicle type, and casualty counts:
if "zip_code" not in data.columns:
    data["zip_code"] = np.nan
if "latitude" not in data.columns:
    data["latitude"] = np.nan
if "longitude" not in data.columns:
    data["longitude"] = np.nan

# axis=1 means go row by row, as opposed to axis=0 (column by column)
data["borough"] = data.apply(get_clean_borough, axis=1)

# fillna replaces missing values
data["factor_group"] = data["contributing_factor_vehicle_1"].fillna("Unspecified").apply(get_factor_group)
data["vehicle_1_group"] = data["vehicle_type_code1"].fillna("Unknown").apply(get_vehicle_group)
data["vehicle_2_group"] = data["vehicle_type_code2"].fillna("Unknown").apply(get_vehicle_group)


# Converts data to numeric value. If the data is missing, changes it to 0.
injured = pd.to_numeric(data["number_of_persons_injured"], errors="coerce").fillna(0)
killed = pd.to_numeric(data["number_of_persons_killed"], errors="coerce").fillna(0)

data["casualty_count"] = injured + killed

# Makes model much more stable to say 10, especially in the case of buses
#   where there is a much higher chance of over 10 casualties.
data["casualty_count"] = data["casualty_count"].clip(lower=0, upper=10)

# Stores as a yes/no in the form of 1 or 0 respectively
data["has_casualty"] = (data["casualty_count"] > 0).astype(int)

# Sort by crash_date, then re-order index (as it will be jumbled) and drop=True throws away old index
data = data.sort_values("crash_date").reset_index(drop=True)

feature_columns = [
    "borough",
    "hour_group",
    "weekday",
    "factor_group",
    "vehicle_1_group",
    "vehicle_2_group",
    "hour",
    "month",
    "is_weekend",
    "is_night",
]

text_columns = [
    "borough",
    "hour_group",
    "weekday",
    "factor_group",
    "vehicle_1_group",
    "vehicle_2_group",
]

# Directing which data goes into model_input
model_input = data[feature_columns]

# get_dummies turns text categories into numeric 0/1 columns (also called "one-hot encoding")
model_input = pd.get_dummies(model_input, columns=text_columns)

# Finds the row number where the split should happen (in this case, it's after the first 3 quarters
#   of the total number of rows)
split_spot = int(len(data) * 0.75)

# iloc is a tool for selecting rows or columns by integer position
#   iloc stands for integer location

# train_input is for everything before split_spot
train_input = model_input.iloc[:split_spot]

# test_input is for everything after split_spot
test_input = model_input.iloc[split_spot]

# Use .copy() for these ones because we will later modify test_data
#   by adding new columns. 
#   It makes it its own independent table.
#   Always use copy when planning to modify the sliced data
train_data = data.iloc[:split_spot].copy()
test_data = data.iloc[split_spot].copy()

print("Training probability model...")

# Give logistic regression up to 1000 attempts/steps to find the best coefficients
probability_model = LogisticRegression(max_iter=1000)

# .fit() trains the model.
#   Learns the relationship between the input features and whether a casualty happened.
probability_model.fit(train_input, train_data["has_casualty"])

print("Training casualty count model...")

# Creates and trains the model that predicts the expected number of casualties per crash.
#   alpha=0.05 controls regularization, a penalty that discourages the model from making its
#   coefficients too extreme (may weigh it WAY too much when it happens only a couple times)
count_model = PoissonRegressor(alpha=0.05, max_iter=500)

# Finds the casualty count we should expect.
count_model.fit(train_input, train_data["casualty_count"])

# Creates new column called 'predicted_casualty_probability'
#   predict_proba predicts the probability of each possible class, only usable once a
#   scikit-learn classification model is trained.
#   [:, 1] keeps only the casualty class (the latter of the result, in the format:
#   [first_num, second_num]. An example of this is: [0.75, 0.25], where 0.25 is 
#   probability of a casualty)
test_data["predicted_casualty_probability"] = probability_model.predict_proba(test_input)[:, 1]

# Predicts expected casualty count by using .predict(), and makes sure none of the predicted
#   values are below 0.0001. This is needed for when we use mean_poisson_deviance
test_data["predicted_casualty_count"] = count_model.predict(test_input)
test_data["predicted_casualty_count"] = test_data["predicted_casualty_count"].clip(lower=0.0001)

# Baseline predictions and evaluation scores
average_train_probability = train_data["has_casualty"].mean()
average_train_count = train_data["casualty_count"].mean()

# Note: A Random Forest is a machine learning model that makes predictions by combining
#   the results of many decision trees.

# nunique counts number of unique values (if it contains both 0 and 1, then
#   we can calculate the AUC. If there is only one value, then AUC can't be calculated).
#   Used for ranking casualty risk
#   Note: Higher is BETTER
if test_data["has_casualty"].nunique() > 1:
    auc_score = roc_auc_score(
        test_data["has_casualty"],
        test_data["predicted_casualty_probability"],
    )
else:
    auc_score = 0

# Used for Probability Error (seeing how good a probability prediction is)
# Lower is BETTER
brier_model = brier_score_loss(
    test_data["has_casualty"],
    test_data["predicted_casualty_probability"]
)

# Used for Baseline Probability Error
#   Takes the average of the Brier Scores among policies (ex. each customer)
#   Lower is BETTER (since we are taking the average of all the model scores)
brier_baseline = brier_score_loss(
    test_data["has_casualty"],
    np.full(len(test_data), average_train_probability),
)

# Count Model Error
#   Measures how well the Poisson model's predicted casualty counts match the
#       real casualty counts.
#   Lower is BETTER
poisson_model_score = mean_poisson_deviance(
    test_data["casualty_count"],
    test_data["predicted_casualty_count"],
)

# Baseline Count Error
#   Gives something for Poisson Model Score to compare against
#   Baseline Score's purpose is to see how good your Poisson Model is: if the baseline
#       is higher than the model, the model is good! If model is higher, it is bad.
#   Lower is BETTER (since we are taking the average of all the model scores)
poisson_baseline_score = mean_poisson_deviance(
    test_data["casualty_count"],
    np.full(len(test_data), max(average_train_count, 0.0001)),
)

# Average Casualty Count Miss
#   Cakcykates the average absolute error
#   Lower is BETTER 
count_error = mean_absolute_error(
    test_data["casualty_count"],
    test_data["predicted_casualty_count"],
)

# Calculating top-risk lift:

risk_cutoff = test_data["predicted_casualty_count"].quantile(0.90)

highest_risk_crashes = test_data[
    test_data["predicted_casualty_count"] >= risk_cutoff
]

# Compares casualty rate in the highest-risk group to the overall test casualty rate
top_lift = highest_risk_crashes["has_casualty"].mean() / test_data["has_casualty"].mean()

# Average predicted casualties per crash * number of portfolio crashes
expected_casualties = test_data["predicted_casualty_count"].mean() * PORTFOLIO_CRASH_COUNT

random_numbers = np.random.default_rng(RANDOM_SEED)

# Simulates the total portfolio casualties many times (SIMULATIONS many times)
simulated_casualties = random_numbers.poisson(expected_casualties, size=SIMULATIONS)

# Estimations for severe portfolio outcomes
#   var = Value at Risk

# 5% of portfolio outcomes had this number of casualties or more
casualty_var_95 = np.quantile(simulated_casualties, 0.95)

# 1% of portfolio outcomes had this number of casualties or more
casualty_var_99 = np.quantile(simulated_casualties, 0.99)

# tvar = Tail Value at Risk
# Takes the average casualty total of casualty_var_95
# 'Among the worst 5% of outcomes, the average total was this number of casualties.
# Tells you about the severity of the tail
casualty_tvar_95 = simulated_casualties[simulated_casualties >= casualty_var_95].mean()


