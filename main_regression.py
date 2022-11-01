import numpy as np
from keras import Input
from numpy.random import seed
import tensorflow as tf
import random

from keras.models import Sequential
from keras.layers import Dense

seed(1)
tf.random.set_seed(1)
tf.config.experimental.enable_op_determinism()
random.seed(2)

import pandas as pd
import matplotlib.pyplot as plt


def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val


if __name__ == '__main__':
    # Read dataset:
    dataset = pd.read_csv('dataset/regression/regression_dataset.csv')
    print(f"There is {len(dataset.index)} instances.")
    print(dataset.head())
    plt.scatter(dataset['x'], dataset['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/regression/dataset.png', bbox_inches='tight')
    plt.show()

    # Split dataset into train and validation:
    train, test = split_dataset(dataset, train_frac=0.8)
    plt.scatter(train['x'], train['y'], c='blue', alpha=0.3)
    plt.scatter(test['x'], test['y'], c='red', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['train set', 'test set'], framealpha=0.3)
    plt.savefig('dataset/regression/dataset_train_test.png', bbox_inches='tight')
    plt.show()

    # Create model:
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    print(model.summary())

    # Train:
    epochs = 2500
    x_train, y_train = train['x'], train['y']
    print(x_train)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_split=0.2)

    # Display loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('dataset/regression/loss.png', bbox_inches='tight')
    plt.show()
    # Display metric:
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mean absolute error (mae)')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('dataset/regression/mae.png', bbox_inches='tight')
    plt.show()

    # Validate:
    test_results = model.evaluate(test['x'], test['y'], verbose=1)
    print(f'Test results - Loss: {test_results[0]} - MAE: {test_results[1]}%')

    # Other display:
    plt.scatter(dataset['x'], dataset['y'])
    plt.plot(x_train.sort_values(), model.predict(x_train.sort_values()), c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/regression/learnt_function.png', bbox_inches='tight')
    plt.show()
