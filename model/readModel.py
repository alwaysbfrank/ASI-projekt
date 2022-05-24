import pandas as pd


def read():
# Read data
    df = pd.read_csv('data/Cobar_cleared.csv')

# Save the data
    df.to_csv('data/Cobar_train.csv', index=False)