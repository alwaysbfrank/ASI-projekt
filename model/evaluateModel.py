import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, datetime


def evaluate():
    df = pd.read_csv('data/AUS_Test.csv')
    y = df['RainTomorrow'].values.reshape(-1, 1)

# Loading the data model
    model = pickle.load(open("model/model_1.2.pkl", 'rb'))

# Prediction
    predictions = model.predict(y)

# Evaluation
    RMSE = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    print('RMSE on test data: ', RMSE)
    print('r2 on test data: ', r2)
