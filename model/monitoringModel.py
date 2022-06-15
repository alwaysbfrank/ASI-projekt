import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import os.path
import glob

from model import readModel


def monitoring(batch_no):
    # Load model
    # model = pickle.load(open("model/model_2022-12-06_15-47-48.pkl", 'rb'))

    # load only last created .pkl file
    list_of_files = glob.glob('model/*.pkl', recursive=True)
    latest_file = max(list_of_files, key=os.path.getctime)
    last_file_timestamp = latest_file.split('model_')[1].split('.pkl')[0]
    model = pickle.load(open(latest_file, 'rb'))


    # Read test data
    test_data = pd.read_csv(f'data/cutted/weatherAUS_{batch_no}.csv')
    test_data.to_csv('data/weatherAUS_current_all.csv', mode='a', index=False, header=False)
    readModel.read()
    all_data = pd.read_csv(f'data/AUS_Prepared.csv')

    X = all_data['Humidity3pm'].values.reshape(-1,1)
    y = all_data['RainTomorrow'].values.reshape(-1,1)

    eval_df = pd.DataFrame(columns=['time_stamp','version','batch','metric','score'])
    eval_df = evaluate_metrics(batch_no, last_file_timestamp, model, y, 'rain', eval_df)
    eval_df = evaluate_metrics(batch_no, last_file_timestamp, model, X, 'humidity', eval_df)

    # Save evaluation to file
    evaluation_file_name = 'data/model_eval.csv'

    if os.path.isfile(evaluation_file_name):
        eval_df.to_csv(evaluation_file_name, mode='a', index=False, header=False)
    else:
        eval_df.to_csv(evaluation_file_name, index=False, header=True)

    return batch_no


def evaluate_metrics(batch_no, last_file_timestamp, model, values, metric_name, eval_df):
    # Predict
    predictions = model.predict(values)
    # Evaluate
    RMSE = np.sqrt(mean_squared_error(values, predictions))
    r2 = r2_score(values, predictions)
    print('RMSE on test data: ', RMSE)
    print('r2 on test data: ', r2)
    # Create the evaluation dataframe
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    eval_df = eval_df.append(
        {'time_stamp': now, 'version': last_file_timestamp, 'batch': batch_no, 'metric': f'{metric_name}_RMSE', 'score': RMSE},
        ignore_index=True)
    eval_df = eval_df.append(
        {'time_stamp': now, 'version': last_file_timestamp, 'batch': batch_no, 'metric': f'{metric_name}_r2', 'score': r2},
        ignore_index=True)
    return eval_df