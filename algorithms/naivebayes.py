import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

def naive_bayes(csv_name: str='vector/set.csv', sample_x: list[list] | None = None):
    '''
    Supervised Learning
    Uses Naive Bayes Algorithm
    Based on the Bayes Probabilities Function

    csv_name is the vectors data
    pos is the position of the last data to train (len (data))
    '''

    dataset = pd.read_csv(f'{csv_name}')
    X_init = dataset.iloc[:, 2:].values
    y_init = dataset.iloc[:, 1].values

    # Training with all the data
    if sample_x is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_init, y_init, test_size=0.5, random_state=1234)

    else:
        sample_x = pd.DataFrame(sample_x)
        X_test = sample_x.iloc[:, 2:].values
        y_test = sample_x.iloc[:, 1].values.tolist()
        y_test=[int(y_i) for y_i in y_test]
        y_test=np.array(y_test)


        X_train = X_init
        y_train = y_init

    # Feature scaling to the training and test set of independent variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training the Naive Bayes model on the training set
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)

    vac = naive_bayes.score(X_train, y_train)
    ac = accuracy_score(y_test, y_pred)

    results = dict()
    results['name'] = "Naive Bayes"
    results['supervised'] = True
    results['score'] = vac
    results['accurency'] = ac

    return y_test, y_pred, results
