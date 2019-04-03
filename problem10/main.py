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

get_x_y =                                                                   \
    lambda filename="wdbc.data":                                            \
        (lambda a: (get_x_data(a), get_y_data(a)))(get_array(get_data(filename)))

def cross_validate(clf, x, y):
    info = sklearn.model_selection.cross_validate(clf, x, cv=10, y=y,
                                                  return_train_score=True)
    print("Average Test Score: {}".format(numpy.mean(info['test_score'])))
    print("Max Test Score: {}".format(max(info['test_score'])))
    print("Average Train Score: {}".format(numpy.mean(info['train_score'])))


def main():
    x, y = get_x_y()

    x = sklearn.preprocessing.normalize(x)
    """
    for kernel in ['linear', 'poly']:
        for gamma in [0.001, 0.5]:
            print("SVM with {} Kernel, gamma={}".format(kernel, gamma))
            cross_validate(svm.SVC(kernel=kernel, gamma=gamma), x, y)
            print()
    """

    for activation in ['logistic', 'tanh']:
        for learning_rate in [0.1, 0.5]:
            print("Neural Network with Activation function {}, "
                  "learning rate {}".format(activation, learning_rate))
            cross_validate(
                sklearn.neural_network.MLPClassifier(
                    activation=activation, learning_rate_init=learning_rate,
                    hidden_layer_sizes=(30)),
                x, y)
            print()

if __name__ == '__main__':
    main()
