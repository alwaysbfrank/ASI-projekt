import pandas as pd
import numpy as np


def read():
    # Read data
    #test
    #df = pd.read_csv('data/weatherAUSTest.csv')
    #not test
    df = pd.read_csv('data/weatherAUS.csv')
    print('Size of weather data frame is :', df.shape)
    # df[0:5]
    df.count().sort_values()
    df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'Date'], axis=1)
    #df.shape
    df = df.dropna(how='any')
    #df.shape
    from scipy import stats
    z = np.abs(stats.zscore(df._get_numeric_data()))
    df = df[(z < 3).all(axis=1)]
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
    df = pd.get_dummies(df, columns=categorical_columns)
    #df.iloc[4:9]
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    #df.iloc[4:10]
    from sklearn.feature_selection import SelectKBest, chi2
    x = df.loc[:, df.columns != 'RainTomorrow']
    y = df[['RainTomorrow']]
    selector = SelectKBest(chi2, k=3)
    selector.fit(x, y)
    #X_new = selector.transform(x)

    df = df[['Humidity3pm', 'Rainfall', 'RainToday', 'RainTomorrow']]

    # Save the data
    #test
    #df.to_csv('data/AUS_Test.csv', index=False)
    #not test
    df.to_csv('data/AUS_Prepared.csv', index=False)
