import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import os.path
import glob


def monitoring():
    # Load model
    # model = pickle.load(open("model/model_2022-12-06_15-47-48.pkl", 'rb'))

    # load only last created .pkl file
    list_of_files = glob.glob('model/*.pkl', recursive=True)
    latest_file = max(list_of_files, key=os.path.getctime)
    last_file_timestamp = latest_file.split('model_')[1].split('.pkl')[0]
    model = pickle.load(open(latest_file, 'rb'))


    # Read test data
    batch_no = 4
    test_data = pd.read_csv('data/AUS_Test.csv')

    X = test_data['Humidity3pm'].values.reshape(-1,1)
    y = test_data['RainTomorrow'].values.reshape(-1,1)

    # Predict
    predictions = model.predict(y)

    # Evaluate
    RMSE = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    print('RMSE on test data: ', RMSE)
    print('r2 on test data: ', r2)

    # Create the evaluation dataframe
    eval_df = pd.DataFrame()

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    eval_df = eval_df.append({'time_stamp':now, 'version': '1.0', 'batch': batch_no, 'metric': 'RMSE', 'score': RMSE}, ignore_index=True)
    eval_df = eval_df.append({'time_stamp':now, 'version': '1.0', 'batch': batch_no, 'metric': 'r2', 'score': r2}, ignore_index=True)

    # Save evaluation to file
    evaluation_file_name = 'data/model_eval_' + last_file_timestamp + '.csv'

    if os.path.isfile(evaluation_file_name):
        eval_df.to_csv(evaluation_file_name, mode='a', index=False, header=False)
    else:
        eval_df.to_csv(evaluation_file_name, index=False)
    return batch_no