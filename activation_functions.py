import matplotlib.pyplot as plt
import numpy as np
from keras.activations import linear, sigmoid, tanh, softplus, softsign, relu, selu, elu

if __name__ == '__main__':
    # Create x:
    min_x = -10
    max_x = 10
    size = 500
    dec = 4
    x = np.unique(np.round(np.random.uniform(low=min_x, high=max_x, size=size), dec))

    functions = [linear, sigmoid, tanh, softplus, softsign, relu, selu, elu]
    functions_name = ['linear', 'sigmoid', 'tanh', 'softplus', 'softsign', 'relu', 'selu', 'elu']
    # Display functions:
    ncols = 4
    nrows = 2
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(len(functions)):
        function = functions[i]
        function_name = functions_name[i]
        ax = axs.flat[i]
        ax.plot(x, function(x))
        ax.set_title(function_name)

    for ax in axs.flat:
        ax.axhline(y=0, clip_on=False, color='gray', linestyle=':')
        ax.axvline(x=0, clip_on=False, color='gray', linestyle=':')
        ax.set(xlabel='x', ylabel='f(x)')
        # ax.label_outer()
    fig.tight_layout()
    plt.savefig('activation_functions.png', bbox_inches='tight')
    plt.show()
