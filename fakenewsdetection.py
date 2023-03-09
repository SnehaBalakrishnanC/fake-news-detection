!unzip /content/drive/MyDrive/FakeNewsDataset/FakeNewsDataset.zip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import keras
from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout

import warnings
warnings.filterwarnings('ignore')


real_data = pd.read_csv('/content/FakeNewsDataset/True.csv')
fake_data = pd.read_csv('/content/FakeNewsDataset/Fake.csv')

real_data.head()

fake_data.head()