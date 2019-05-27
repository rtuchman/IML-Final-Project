import pandas as pd
import numpy as np
from feature_selector import FeatureSelector
import warnings
warnings.filterwarnings('ignore')


class PreprocessingPipeline():

    def __init__(self, train, test):
        self._df_train = pd.read_csv(train)
        self._df_test = pd.read_csv(test)

    def handle_missing_data(self, test=False):
        """Impute missing values.

          Columns of dtype object are imputed with the most frequent value
          in column.

          Columns of other types are imputed with mean of column.

          """
        if test:
            X = pd.DataFrame(self._df_test)
            fill_na_test = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                                                     else X[c].mean() for c in X], index=X.columns)

            self._fill_na_test_dict = {str(k): v for k, v in fill_na_test.items()}
            self._df_test.fillna(self._fill_na_test_dict, inplace=True)
        else:
            X = pd.DataFrame(self._df_train.drop(['label'], axis=1).values)
            fill_na_train = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                                                     else X[c].mean() for c in X], index=X.columns)

            self._fill_na_train_dict = {str(k): v for k, v in fill_na_train.items()}
            self._df_train.fillna(self._fill_na_train_dict, inplace=True)



    def handle_unknown_features(self, test=False):
        """Nullify samples in which df[3] == 'unknown'
        """
        if test:
            self._df_test['3'].replace('unknown', np.NaN, inplace=True)
        else:
            self._df_train['3'].replace('unknown', np.NaN, inplace=True)



    def encode_categorical(self, test=False):
        if test:
            # find categorial colimns
            cols = self._df_test.columns
            num_cols = self._df_test._get_numeric_data().columns
            categorical_indices = list(set(cols) - set(num_cols))

            # create dummis
            dummies_list = [pd.get_dummies(self._df_test[idx], prefix='f{}'.format(idx)) for idx in categorical_indices]
            frames = dummies_list + [self._df_test]
            self._df_test = pd.concat(frames, axis=1)
            self._df_test = self._df_test.drop(columns=categorical_indices)
        else:
            # find categorial colimns
            cols = self._df_train.columns
            num_cols = self._df_train._get_numeric_data().columns
            categorical_indices = list(set(cols) - set(num_cols))

            # create dummis
            dummies_list = [pd.get_dummies(self._df_train[idx], prefix='f{}'.format(idx)) for idx in categorical_indices]
            frames = dummies_list + [self._df_train]
            self._df_train = pd.concat(frames, axis=1)
            self._df_train = self._df_train.drop(columns=categorical_indices)
        return


if __name__ == '__main__':
    dp = PreprocessingPipeline(train='train.csv', test='test_without_target.csv')
    dp.handle_missing_data()
    dp.handle_missing_data(test=True)
    dp.handle_unknown_features()
    dp.handle_unknown_features(test=True)
    dp.encode_categorical()
    dp.encode_categorical(test=True)
