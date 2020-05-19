import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import time
import warnings
import os

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(__file__)


def preprocess_data():
    global train, all_data, le
    test = pd.read_csv(os.path.join(script_dir, "test.csv"))
    train = pd.read_csv(os.path.join(script_dir, "train.csv"))
    # Merge train and test, exclude overlap
    dates_overlap = ['2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25',
                     '2020-03-26', '2020-03-27']
    train2 = train.loc[~train['Date'].isin(dates_overlap)]
    all_data = pd.concat([train2, test], axis=0, sort=False)
    # Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11
    all_data.loc[all_data['Date'] >= '2020-03-19', 'ConfirmedCases'] = np.nan
    all_data.loc[all_data['Date'] >= '2020-03-19', 'Fatalities'] = np.nan
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    # Create date columns
    le = preprocessing.LabelEncoder()
    all_data['Day_num'] = le.fit_transform(all_data.Date)
    all_data['Day'] = all_data['Date'].dt.day
    all_data['Month'] = all_data['Date'].dt.month
    all_data['Year'] = all_data['Date'].dt.year
    # Fill null values given that we merged train-test datasets
    all_data['Province_State'].fillna("None", inplace=True)
    all_data['ConfirmedCases'].fillna(0, inplace=True)
    all_data['Fatalities'].fillna(0, inplace=True)
    all_data['Id'].fillna(-1, inplace=True)
    all_data['ForecastId'].fillna(-1, inplace=True)
    ts = time.time()
    all_data = calculate_lag(all_data, range(1, 7), 'ConfirmedCases')
    all_data = calculate_lag(all_data, range(1, 7), 'Fatalities')
    all_data = calculate_trend(all_data, range(1, 7), 'ConfirmedCases')
    all_data = calculate_trend(all_data, range(1, 7), 'Fatalities')
    all_data.replace([np.inf, -np.inf], 0, inplace=True)
    all_data.fillna(0, inplace=True)
    # print("Time spent: ", time.time()-ts)
    preprocess_world_population()
    all_data = all_data.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)',
                              how='left')
    all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[
        ['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)

    # Label encode countries and provinces. Save dictionary for exploration purposes
    all_data.drop('Country (or dependency)', inplace=True, axis=1)
    all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])
    number_c = all_data['Country_Region']
    countries = le.inverse_transform(all_data['Country_Region'])
    country_dict = dict(zip(countries, number_c))
    all_data['Province_State'] = le.fit_transform(all_data['Province_State'])
    number_p = all_data['Province_State']
    province = le.inverse_transform(all_data['Province_State'])
    province_dict = dict(zip(province, number_p))
    return country_dict, province_dict


def calculate_trend(df, lag_list, column):
    for lag in lag_list:
        trend_column_lag = "Trend_" + column + "_" + str(lag)
        temp = df[column]
        temp2 = df[column].shift(lag, fill_value=-999)
        df[trend_column_lag] = (df[column] - df[column].shift(lag, fill_value=-999)) / df[column].shift(lag,
                                                                                                        fill_value=0)
    return df


def calculate_lag(df, lag_list, column):
    for lag in lag_list:
        column_lag = column + "_" + str(lag)
        df[column_lag] = df[column].shift(lag, fill_value=0)
    return df


def preprocess_world_population():
    global world_population
    world_population = pd.read_csv(os.path.join(script_dir, "population_by_country_2020.csv"))
    # Select desired columns and rename some of them
    world_population = world_population[
        ['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age',
         'Urban Pop %']]
    world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age',
                                'Urban Pop']
    # Replace United States by US
    world_population.loc[
        world_population['Country (or dependency)'] == 'United States', 'Country (or dependency)'] = 'US'
    # Remove the % character from Urban Pop values
    world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')
    # Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int
    world_population.loc[world_population['Urban Pop'] == 'N.A.', 'Urban Pop'] = int(
        world_population.loc[world_population['Urban Pop'] != 'N.A.', 'Urban Pop'].mode()[0])
    world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')
    world_population.loc[world_population['Med Age'] == 'N.A.', 'Med Age'] = int(
        world_population.loc[world_population['Med Age'] != 'N.A.', 'Med Age'].mode()[0])
    world_population['Med Age'] = world_population['Med Age'].astype('int16')
    return world_population


def maybe_used_later():
    global data
    data = all_data.copy()
    features = ['Id', 'ForecastId', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities',
                'Day_num', 'Day', 'Month', 'Year']
    data = data[features]
    # Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
    data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')
    data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))
    # Replace infinites
    data.replace([np.inf, -np.inf], 0, inplace=True)
    # display(data)
    x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)


# Split data into train/test
def split_data(data):
    # Train set
    x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)
    y_train_1 = data[data.ForecastId == -1]['ConfirmedCases']
    y_train_2 = data[data.ForecastId == -1]['Fatalities']

    # Test set
    x_test = data[data.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    # Clean Id columns and keep ForecastId as index
    x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    return x_train, y_train_1, y_train_2, x_test


# Linear regression model
def lin_reg(X_train, Y_train, X_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    return regr, y_pred


# Submission function
def get_submission(df, target1, target2):
    prediction_1 = df[target1]
    prediction_2 = df[target2]

    # Submit predictions
    prediction_1 = [int(item) for item in list(map(round, prediction_1))]
    prediction_2 = [int(item) for item in list(map(round, prediction_2))]

    submission = pd.DataFrame({
        "ForecastId": df['ForecastId'].astype('int32'),
        "ConfirmedCases": prediction_1,
        "Fatalities": prediction_2
    })
    submission.to_csv('submission.csv', index=False)

    # New split function, for one forecast day


def split_data_one_day(data, d):
    # Train
    x_train = data[data.Day_num < d]
    y_train_1 = x_train.ConfirmedCases
    y_train_2 = x_train.Fatalities
    x_train.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    # Test
    x_test = data[data.Day_num == d]
    x_test.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    # Clean Id columns and keep ForecastId as index
    x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    return x_train, y_train_1, y_train_2, x_test


def plot_real_vs_prediction_country(data, train, country_name, day_start, dates_list):
    # Select predictions from March 1st to March 25th
    predicted_data = data.loc[
        (data['Day_num'].isin(list(range(day_start, day_start + len(dates_list)))))].ConfirmedCases
    real_data = train.loc[(train['Country_Region'] == country_name) & (train['Date'].isin(dates_list))][
        'ConfirmedCases']

    print((predicted_data))
    dates_list_num = list(range(0, len(dates_list)))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(dates_list_num, np.exp(predicted_data))
    ax1.plot(dates_list_num, real_data)
    ax1.axvline(17, linewidth=2, ls=':', color='grey', alpha=0.5)
    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax1.set_xlabel("Day count (starting on March 1st)")
    ax1.set_ylabel("Confirmed Cases")

    ax2.plot(dates_list_num, predicted_data)
    ax2.plot(dates_list_num, np.log(real_data))
    ax2.axvline(17, linewidth=2, ls=':', color='grey', alpha=0.5)
    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax2.set_xlabel("Day count (starting on March 1st)")
    ax2.set_ylabel("Log Confirmed Cases")

    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for " + country_name))
    plt.savefig('/Users/ariel-pc/Desktop/Projects/a')
    plt.show()


def plot_real_vs_prediction_country_fatalities(data, train, country_name, day_start, dates_list):
    # Select predictions from March 1st to March 25th
    predicted_data = data.loc[
        (data['Day_num'].isin(list(range(day_start, day_start + len(dates_list)))))].Fatalities
    real_data = train.loc[(train['Country_Region'] == country_name) & (train['Date'].isin(dates_list))][
        'Fatalities']
    print(np.exp(predicted_data))
    dates_list_num = list(range(0, len(dates_list)))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(dates_list_num, np.exp(predicted_data))
    ax1.plot(dates_list_num, real_data)
    ax1.axvline(17, linewidth=2, ls=':', color='grey', alpha=0.5)
    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax1.set_xlabel("Day count (starting on March 1st)")
    ax1.set_ylabel("Fatalities Cases")

    ax2.plot(dates_list_num, predicted_data)
    ax2.plot(dates_list_num, np.log(real_data))
    ax2.axvline(17, linewidth=2, ls=':', color='grey', alpha=0.5)
    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax2.set_xlabel("Day count (starting on March 1st)")
    ax2.set_ylabel("Log Fatalities Cases")

    plt.suptitle(("Fatalities predictions based on Log-Lineal Regression for " + country_name))
    plt.show()


def lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict):
    ts = time.time()

    # Filter country and features from all_data (dataset without data leaking)
    data = all_data.copy()
    features = ['Id', 'Province_State', 'Country_Region',
                'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num']
    data = data[features]

    # Select country an data start (all days)
    data = data[data['Country_Region'] == country_dict[country_name]]
    data = data.loc[data['Day_num'] >= day_start]

    # Lags
    data = calculate_lag(data, range(1, lag_size), 'ConfirmedCases')
    data = calculate_lag(data, range(1, 8), 'Fatalities')

    filter_col_confirmed = [col for col in data if col.startswith('Confirmed')]
    filter_col_fatalities = [col for col in data if col.startswith('Fataliti')]
    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)

    # Apply log transformation
    data[filter_col] = data[filter_col].apply(lambda x: np.log(x))
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)

    # Start/end of forecast
    start_fcst = all_data[all_data['Id'] == -1].Day_num.min()
    end_fcst = all_data[all_data['Id'] == -1].Day_num.max()

    for d in list(range(start_fcst, end_fcst + 1)):
        # print("Predicting day:" + str(d) + "\n")
        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d)
        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)
        data.loc[(data['Country_Region'] == country_dict[country_name])
                 & (data['Day_num'] == d), 'ConfirmedCases'] = pred_1[0]
        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)
        data.loc[(data['Country_Region'] == country_dict[country_name])
                 & (data['Day_num'] == d), 'Fatalities'] = pred_2[0]

        # Recompute lags
        data = calculate_lag(data, range(1, lag_size), 'ConfirmedCases')
        data = calculate_lag(data, range(1, 8), 'Fatalities')
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

    return data


def grid_search(X_train, Y_train_1, X_valid, Y_valid_1):
    ts = time.time()

    xgb1 = xgb.XGBRegressor()
    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'learning_rate': [.07, .01, .005],  # so called `eta` value
                  'max_depth': [4, 5, 6, 7],
                  'min_child_weight': [4, 5, 6, 7],
                  'silent': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500]}

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv=3,
                            n_jobs=5,
                            verbose=True)

    xgb_grid.fit(X_train, Y_train_1)

    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)

    Y_pred = xgb_grid.predict(X_valid)
    print(Y_pred, Y_valid_1)

    print("Time spent: ", ts - time.time())


if __name__ == "__main__":
    country_dict, province_dict = preprocess_data()
    country_name = 'Spain'
    day_start = 35
    lag_size = 30
    data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)
    # Select train (real) data from March 1 to March 22nd
    dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07',
                  '2020-03-08', '2020-03-09',
                  '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14', '2020-03-15', '2020-03-16',
                  '2020-03-17', '2020-03-18',
                  '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25',
                  '2020-03-26', '2020-03-27']
    # maybe_used_later()
    plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)
    #plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
