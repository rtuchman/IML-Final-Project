import pandas as pd
import numpy as np
from feature_selector import FeatureSelector
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessing():

    def __init__(self):
        self._df_train = pd.read_csv("train.csv")

    def handle_missing_data(self):
        """Impute missing values.

          Columns of dtype object are imputed with the most frequent value
          in column.

          Columns of other types are imputed with mean of column.

          """
        y = pd.DataFrame(self._df_train.label.values)
        X = pd.DataFrame(self._df_train.drop(['label'], axis=1).values)
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                             else X[c].mean() for c in X], index=X.columns)

        self._df_train = pd.concat([X.fillna(self.fill), y], axis=1)
        print('success')
        return


    def lose_unknown(self):
        """Throw away samples in which df[3] == 'unknown'

        """
        self._df_train = self._df_train[self._df_train[3] != 'unknown']

        return


if __name__ == '__main__':
    dp = DataPreprocessing()
    dp.handle_missing_data()
    dp.lose_unknown()

