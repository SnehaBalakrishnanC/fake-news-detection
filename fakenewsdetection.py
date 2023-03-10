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

real_data['target'] = 1
fake_data['target'] = 0 

real_data.tail()

fake_data.tail()

data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
data.head()

data.isnull().sum()

print(data["target"].value_counts())
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(data.target, ax=ax[0], palette=["red", "green"])
g1.set_title("Count of real and fake data")
g1.set_ylabel("Count")
g1.set_xlabel("Target")
g2 = plt.pie(data["target"].value_counts().values,explode=[0,0],labels=data.target.value_counts().index, autopct='%1.1f%%',colors=['red','green'])
fig.show()


print(data.subject.value_counts())
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="subject",  hue='target', data=data, palette=["red", "green"])
plt.title("Distribution of The Subject According to Real and Fake Data")


data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']
data.head()

first_text = data.text[10]
first_text

pip install bs4

from bs4 import BeautifulSoup



soup = BeautifulSoup(first_text, "html.parser")
first_text = soup.get_text()
first_text


first_text = re.sub('\[[^]]*\]', ' ', first_text)
first_text = re.sub('[^a-zA-Z]',' ',first_text)  # replaces special characters with spaces
first_text = first_text.lower() # Converting all from uppercase to lowercase
first_text


nltk.download("stopwords")
nltk.download('punkt')
from nltk.corpus import stopwords  

# we can use tokenizer instead of split
first_text = nltk.word_tokenize(first_text)

first_text = [ word for word in first_text if not word in set(stopwords.words("english"))]