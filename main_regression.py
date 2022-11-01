from keras import Input
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
tf.config.experimental.enable_op_determinism()
import random
random.seed(2)


def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val


if __name__ == '__main__':
    # Read dataset:
    dataset = pd.read_csv('dataset/regression/train.csv')
    print(f"There is {len(dataset.index)} instances.")
    print(dataset.head())
    plt.scatter(dataset['x'], dataset['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/regression/dataset.png', bbox_inches='tight')
    plt.show()

    # Split dataset into train and validation:
    train, validation = split_dataset(dataset, train_frac=0.7)
    plt.scatter(train['x'], train['y'], c='blue', alpha=0.4)
    plt.scatter(validation['x'], validation['y'], c='red', alpha=0.4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['train set', 'validation set'], framealpha=0.3)
    plt.savefig('dataset/regression/dataset_split.png', bbox_inches='tight')
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
    loss = 'mse'
    metric = 'mae'
    epochs = 2500
    x_train, y_train = train['x'], train['y']
    x_val, y_val = validation['x'], validation['y']
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_data=(x_val, y_val))

    # Display loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('dataset/regression/loss.png', bbox_inches='tight')
    plt.show()
    # Display metric:
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(f'dataset/regression/{metric}.png', bbox_inches='tight')
    plt.show()

    # Evaluate on test set:
    test = pd.read_csv('dataset/regression/test.csv')
    test_results = model.evaluate(test['x'], test['y'], verbose=1)
    print(f'Test set: - loss: {test_results[0]} - {metric}: {test_results[1]}')

    # Display function:
    plt.scatter(dataset['x'], dataset['y'])
    plt.plot(x_train.sort_values(), model.predict(x_train.sort_values()), c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/regression/learnt_function.png', bbox_inches='tight')
    plt.show()
