import tensorflow as tf
#load libraries
import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the file "data.csv" and print the contents.
#!cat data/data.csv
data = pd.read_csv('data/data.csv', index_col=False,)


#NoteBook 1
# Id column is redundant and not useful, we want to drop it
data.drop('id', axis =1, inplace=True)
#data.drop('Unnamed: 0', axis=1, inplace=True)
data.head(2)
data.shape
data.info()
data.get_dtype_counts()
#check for missing variables
data.isnull().any()
data.diagnosis.unique()
#save the cleaner version of dataframe for future analyis
data.to_csv('data/clean-data.csv')



#Notebook 2
