#!/usr/bin/python3

#
# Can run directly calling `python3 neural_network.py [csvpath]`
# 
# Raw:      91.36%, 4m 3s
# Balanced: 77.23%, 1m 38s
#

"""
Se setean las semillas de randoms de numpy y tensorflow para tener consistencia entre corridas
"""

import sys 
import random
import numpy as np
np.random.seed(91218) # Set np seed for consistent results across runs

import tensorflow as tf
tf.set_random_seed(91218)

from tensorflow import keras
from sklearn.model_selection import train_test_split


DEAFULT_CSVPATH = '../data/raw.csv'


"""## Extracción y procesamiento de datos
Se definen las funciones para extracción y procesamiento del set de datos.
Se descarta la columna de IDs y se normalizan el resto de las columnas. No se encontraron registros nulos
"""

def show_progress(done, total, size=100):
    completed = int(1.0 * size * done / total)
    bar = '[{}{}] {}/{}'.format(
        '=' * completed, 
        '.' * (size - completed),
        done, 
        total
    )
    print(bar, end='\r')
    if done >= total: print('')

  
def process_row(row):
  # Remove ID_code column
  return list(map(lambda value: float(value), row[1:]))
  

def get_minmax(minmax, value):
  if not minmax: minmax = {'min': float('inf'), 'max': float('-inf')}
  return {'min': min(minmax['min'], value), 'max': max(minmax['max'], value)}

def extract_data(csvfile, balanced=False):
  print('Reading csv...')

  rows = csvfile.read().splitlines()
  
  # Remove headaer and parse rows
  header = rows[0].split(',')
  rows = rows[1:]
  rawdata = [row.split(',') for row in rows]
  rowscount = len(rawdata)

  results = []
  minmaxs = {}
  i = 0
  target1count = 0
  print('Processing data...')
  for row in rawdata:
    i += 1
    show_progress(i, rowscount)
    processed = process_row(row)
    results.append(processed)  
    minmaxs = {i: get_minmax(minmaxs.setdefault(i, None), processed[i]) for i in range(len(processed))}
    if processed[0] == 1.0: target1count += 1

  if balanced:
    print('Balancing data...')
    # We assume there are less 1's than 0's
    limit = len(results) - 2 * target1count
    results.sort(key=lambda row: row[0]) # Sort by target
    results = list(map(lambda pair: pair[1], filter(lambda pair: pair[0] >= limit, enumerate(results))))
    results.sort(key=lambda row: random.random())

  print('Normalizing rows...')
  results = list(map(lambda row: normalize_row(row, minmaxs), results))
  
  return results


def normalize_row(row, minmaxs):
  return [(value-minmaxs[i]['min'])/(minmaxs[i]['max']-minmaxs[i]['min']) for i, value in enumerate(row)]


def categorical_train(x, y):
  max_x = int(len(x) * 0.67) # 33% used for test
  max_y = int(len(y) * 0.67) # 33% used for test
  x_train, x_test, y_train, y_test = x[:max_x], x[max_x:], y[:max_y], y[max_y:]
  y_train = keras.utils.to_categorical(y_train, num_classes=2)
  y_test = keras.utils.to_categorical(y_test, num_classes=2)
  return x_train, x_test, y_train, y_test


def main():
  """Carga de datos
  Se cargan los datos sin categorizar y categorizados.
  Se toman las primeras 8 columnas como features y la ultima como output.
  """


  csvpath = sys.argv[1] if len(sys.argv) > 1 else DEAFULT_CSVPATH
  csv = open(csvpath, 'r')

  normal_data = extract_data(csv)


  """## Entrenamiento de la primera red
  Se crea la primera red usando datos sin categorizar. Como funcion de loss se utiliza el error cuadrático medio
  """

  dataset = np.array(normal_data)
  x = dataset[:, 1:]
  y = dataset[:, 0]

  print('Separating train & test...')
  X_train, X_test, y_train, y_test = categorical_train(x, y)

  input_dim = len(x[0])
  model = keras.Sequential([
      keras.layers.Dense(5, input_dim=input_dim, activation='relu'),
      keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

  print('Fitting model...')
  model.fit(X_train, y_train, epochs=100, batch_size=100, shuffle=False)

  """## Ejecución de la primera red
  Ejecutamos la primera red con los datos no categorizados
  """

  print('Evaluating model...')
  test_loss, test_acc = model.evaluate(X_test, y_test)
  print('Test accuracy:', round(test_acc*100, 2), '%')


if __name__ == "__main__":
  main()
