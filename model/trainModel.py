import pandas as pd
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


def train_new():
    df = pd.read_csv('data/AUS_Prepared.csv')

    x = df[['Humidity3pm']]
    y = df[['RainTomorrow']]

    t0 = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    clf_logreg = LogisticRegression(random_state=0)
    clf_logreg.fit(x_train, y_train)
    y_pred = clf_logreg.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)
    print('Time taken :', time.time() - t0)
    print('...Exporting the model...')

    now = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    pickle.dump(clf_logreg, open('model/model_'+now+'.pkl', 'wb'))
