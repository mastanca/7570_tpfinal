import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

import sys


DEFAULT_CSV = "data/raw.csv"


# Lectura de datos
csvfile = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
dataset = pd.read_csv(csvfile)

X = dataset.drop(['ID_code', 'target'], axis=1)
y = dataset.target

# Split en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Training features shape: ', X_train.shape)
X_train.head()

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Sklearn wants the labels as one-dimensional vectors
y_train = np.array(y_train).reshape((-1,))
y_test = np.array(y_test).reshape((-1,))


# Modelado
tpot = TPOTClassifier(scoring="accuracy",
                      generations=5,
                      cv=5,
                      max_time_mins=2,
                      n_jobs=-1,
                      random_state=42,
                      verbosity=2,
                      use_dask=True)

tpot.fit(X_train, y_train)

print("Accuracy: " + str(round(tpot.score(X_test, y_test),2)*100) + "%")
