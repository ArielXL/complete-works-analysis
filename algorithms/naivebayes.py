import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


def naive_bayes(csv_name: str, pos: int | None):
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
    if pos is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_init, y_init, test_size=0.5, random_state=1234)

    else:
        X_train = X_init[:pos]
        y_train = y_init[:pos]

        X_test = X_init[pos:]
        y_test = y_init[pos:]

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
    # predictions = naive_bayes.predict(X_test)
    # p = classification_report(y_test, predictions)

    results = dict()
    results['name'] = "Naive Bayes"
    results['supervised'] = True
    results['score'] = vac
    results['accurency'] = ac

    return y_test, y_pred, results
