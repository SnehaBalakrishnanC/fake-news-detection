# -*- coding: utf-8 -*-
"""avani.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AzVueNORa3ukb9moIgwahX0To5TY9sBp
"""

fake_news=fake_news[~fake_news.date.str.contains("http")]
fake_news=fake_news[~fake_news.date.str.contains("HOST")]



fake_news['date'] = pd.to_datetime(fake_news['date'])
true_news['date'] = pd.to_datetime(true_news['date'])

clean_news=news_dataset.copy()

