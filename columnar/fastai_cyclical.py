from fastai import *          
from fastai.tabular import *  
from fastai.docs import *     
from data import prepare_data
import numpy as np
import pandas as pd

x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data(one_hot=False)
train_df = pd.DataFrame(x_train)
train_df['target'] = np.expand_dims(y_train.astype(int).replace(-1, 2), axis=1)
valid_df = pd.DataFrame(x_valid)
valid_df['target'] = np.expand_dims(y_valid.astype(int).replace(-1, 2), axis=1)

dep_var = 'target'
data = tabular_data_from_df('abc_merged', train_df, valid_df, dep_var, cont_names=list(range(34)), cat_names=[])
learn = get_tabular_learner(data, layers=[200, 100], metrics=accuracy)
learn.fit_one_cycle(100, 1e-2)
