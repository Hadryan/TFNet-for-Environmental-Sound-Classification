# Cross-fold lists generator for ESC-50 Dataset.

import pandas as pd

def read_meta(metadata_path):
    df = pd.read_csv(metadata_path, sep=',')
    return df

def make_folds(df, test_fold, train_fold):
    test_path = 'fold'+ str(test_fold)+ '_test.csv'
    train_path = 'fold'+ str(test_fold)+ '_train.csv'
    data = df
    data_test = data[data['fold'].isin([test_fold])]
    fp_test = open(test_path, 'w')
    fp_test.write(data_test.to_csv(header=True, index=False))
    
    data_train = data[data['fold'].isin(train_fold)]
    fp_train = open(train_path, 'w')
    fp_train.write(data_train.to_csv(header=True, index=False))
  

if __name__ == '__main__':
    # You need to modify this path to your downloaded dataset directory
    metadata_path = '/.../ESC-50/meta/esc50.csv'
    df = read_meta(metadata_path)
    make_folds(df, 1, [2,3,4,5])
    make_folds(df, 2, [1,3,4,5])
    make_folds(df, 3, [1,2,4,5])
    make_folds(df, 4, [1,2,3,5])
    make_folds(df, 5, [1,2,3,4])