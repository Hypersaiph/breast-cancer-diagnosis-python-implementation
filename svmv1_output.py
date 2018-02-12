
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "data/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
#%matplotlib inline
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv("data/clean-data.csv",header = 0)
#print(df.head())
#df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 0',axis=1,inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
#print(df.head())
data_df= list(df.columns[1:31])
data_df_main = df.loc[:,data_df]
#print(data_df_main)


X = data_df_main
y = df['diagnosis']

svm_model = SVC(verbose=True)

parameters = [
    {'C': [1, 10, 100, 1000],
     'kernel': ['linear']
     },
]

filename = 'finalized_model_jl.sav'
loaded_model = joblib.load(filename)
result = loaded_model.score(X, y)
print(loaded_model.best_score_)
print(result)

Xtest = data_df_main.loc[135:135,data_df]

Ypredict = loaded_model.predict(Xtest)
print(Ypredict)