import pandas as pd
import numpy as np
import shutil
from sklearn import preprocessing


def process_data():
    df = pd.read_csv('data/weatherAUS_current_all.csv')
    df.count().sort_values()
    df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'Date'], axis=1)
    df = df.dropna(how='any')
    from scipy import stats
    z = np.abs(stats.zscore(df._get_numeric_data()))
    df = df[(z < 3).all(axis=1)]
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
    df = pd.get_dummies(df, columns=categorical_columns)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    df = df[['Humidity3pm', 'Rainfall', 'RainToday', 'RainTomorrow']]
    df.to_csv('data/AUS_Prepared.csv', index=False)


def initialize():
    shutil.copyfile('data/weatherAUS_starter.csv', 'data/weatherAUS_current_all.csv')