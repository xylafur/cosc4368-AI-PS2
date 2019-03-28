import pandas as pd
import numpy

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.neural_network

from sklearn import svm


get_data = lambda filename="wdbc.data": pd.read_csv(filename)

get_array = lambda pd_data: pd_data.to_numpy()

get_x_data = lambda _2d_array: numpy.array([numpy.array(e[2:]) for e in _2d_array])
get_y_data = lambda _2d_array: numpy.array([e[1] for e in _2d_array])

get_x_y = lambda: (lambda a: (get_x_data(a), get_y_data(a)))(get_array(get_data()))

def cross_validate(clf, x, y):
    info = sklearn.model_selection.cross_validate(clf, x, cv=10, y=y)
    for key, val in info.items():
        print("    {} = {}".format(key, val))



def main():
    x, y = get_x_y()

    x = sklearn.preprocessing.normalize(x)

    for kernel in ['linear', 'poly']:
        print("SVM with {} Kernel, gamma=0.001".format(kernel))
        cross_validate(svm.SVC(kernel=kernel, gamma=0.001), x, y)
        print()

    for activation in ['logistic', 'tanh']:
        print("Neural Network with Activation function {}".format(activation))
        cross_validate(
            sklearn.neural_network.MLPClassifier(activation=activation,
                                                 learning_rate_init=0.1),
            x, y)
        print()






if __name__ == '__main__':
    main()

