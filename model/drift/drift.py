import pandas as pd
import numpy as np


def should_retrain():
    #todo seperate metrics for humidity and rain
    eval_results = pd.read_csv('data/model_eval.csv', parse_dates=['time_stamp'], dayfirst=True)
    last_rmse, rest_rmse = get_metric('RMSE', eval_results)
    has_rmse_drifted = has_drifted(last_rmse, rest_rmse)
    if has_rmse_drifted:
        print(f'last RMSE: {last_rmse} suggests drift, should retrain')
        return True

    last_r2, rest_r2 = get_metric('r2', eval_results)
    has_r2_drifted = has_drifted(last_r2, rest_r2, False)

    if has_r2_drifted:
        print(f'last r2: {last_r2} suggests drift, should retrain')
    return has_r2_drifted


def get_metric(metric, csv):
    last_evaluation_timestamp = csv['time_stamp'].max()
    all = csv[csv['metric']==metric]
    last = all[all['time_stamp']==last_evaluation_timestamp]['score'].values[0]
    other = all[all['time_stamp']!=last_evaluation_timestamp]['score'].values
    return last, other


def has_drifted(tested, rest, upwards = True):
    parameter = 1
    if not upwards:
        parameter = -1
    test_against = np.mean(rest) + 2 * parameter * np.std(rest)
    if upwards:
        return tested > test_against
    else:
        return tested < test_against
