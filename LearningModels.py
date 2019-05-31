import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.metrics import confusion_matrix
from keras import optimizers
import warnings
warnings.filterwarnings("ignore")
import pandas as pd




class RunModels():

    def __init__(self, dataset):

            X = dataset.iloc[:, 1:-1].values
            y = dataset.iloc[:, -1].values
            self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X, y,
                                                                                                test_size=0.25,
                                                                                                random_state=0)
            self.input_dim = X.shape[1]


    def ANN(self, layers=(50, 50, 50), weights_init='glorot_uniform', activation='relu', dropout_rate=0.1,
            optimizer='adam',  lr=0.00095, beta_1=0.9299, beta_2=0.995):

        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dropout(dropout_rate))
        classifier.add(Dense(input_dim=self.input_dim, units=layers[0],
                                  kernel_initializer=weights_init, activation=activation))

        # Adding the hidden layers
        for neurons in layers[1:]:
            classifier.add(Dropout(dropout_rate))
            classifier.add(Dense(units=neurons, kernel_initializer=weights_init, activation=activation))

        # Adding the output layer
        classifier.add(Dense(units=1, kernel_initializer=weights_init, activation='sigmoid'))

        # Compiling the ANN
        adam = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

        return classifier

    def create_confusion_matrix(self, y, y_pred):

        # Making the Confusion Matrix
        self.cm = confusion_matrix(y, y_pred)

    def plot_confusion_matrix(self, title='Confusion matrix', cmap=None, normalize=True):

        cm = self.cm

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        target_names = [0, 1]
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")#

        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))












