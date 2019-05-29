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


    def ANN(self, layers=(50, 50), weights_init='glorot-uniform', activation='relu', drop_amount=(0.05, 0.25),
            adam_lr=0.001, adam_beta_1=0.9, adam_beta_2=0.999):
        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dropout(drop_amount[0]))
        classifier.add(Dense(input_dim=self.input_dim, units=layers[0],
                                  kernel_initializer=weights_init, activation=activation))

        # Adding the hidden layers
        for l in layers[1:]:
            classifier.add(Dropout(drop_amount[1]))
            classifier.add(Dense(units=layers[l], kernel_initializer=weights_init, activation=activation))

        # Adding the output layer
        classifier.add(Dense(units=2, kernel_initializer=weights_init, activation='softmax'))

        # Compiling the ANN
        adam = optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2)  # faster
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

        return classifier

    def create_confusion_matrix(self, y, y_pred):

        # Making the Confusion Matrix
        y_predict_non_category = [np.argmax(t) for t in y_pred]
        self.cm = confusion_matrix(y, y_predict_non_category)

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

if __name__ == '__main__':
    low_dim_processed_train = pd.read_csv('low_dim_processed_train')
    LM_1 = LearningModels(dataset=low_dim_processed_train)
    NB_clf = LM_1.NaiveBayes()
    NB_clf.fit(LM_1.X_train, LM_1.y_train)
    y_pred = NB_clf.predict_proba(LM_1.X_validation)
    LM_1.create_confusion_matrix(LM_1.y_validation, y_pred)
    LM_1.plot_confusion_matrix()











