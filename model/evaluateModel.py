import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, datetime


def evaluate():
    df = pd.read_csv('data/AUS_Test.csv')
    X = df['Humidity3pm'].values.reshape(-1, 1)
    y = df['RainTomorrow'].values.reshape(-1, 1)
    from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Load model
    model = pickle.load(open("model/model_1.2.pkl", 'rb'))

    # Read test data
    # test_data = pd.read_csv("data/weatherAUS.csv")
    test_data = train_test_split(X, y, test_size=0.25)

    # X = test_data['x'].values.reshape(-1,1)
    # y = test_data['y'].values.reshape(-1,1)

    # Predict
    predictions = model.predict(y)

    # Evaluate
    RMSE = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    print('RMSE on test data: ', RMSE)
    print('r2 on test data: ', r2)
