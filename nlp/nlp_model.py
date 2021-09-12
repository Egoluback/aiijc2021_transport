import numpy as np
import pandas as pd
import re

from collections import Counter
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
import catboost as ctb
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
from sklearn import utils

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from pymorphy2 import MorphAnalyzer
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

import gensim
import gensim.downloader as api
from gensim.test.utils import get_tmpfile
from gensim.models import FastText, KeyedVectors
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')



class Model:
  def __init__(self):
    self.model = None
    self.cars_vectorizer = Doc2Vec(min_count=1, vector_size=8, window=2, workers=-1, seed=42)
    self.comm_model = LogisticRegression(random_state=42, max_iter=1000)
    self.cars_clusterer = KMeans(n_clusters=4, random_state=42, n_init=50, max_iter=300)
    self.text_vectorizer = Doc2Vec(min_count=1, vector_size=30, window=5, workers=-1, seed=42)
    
    self.standart_comments = ['Больше нечего сказать'.lower(), 'Да'.lower(), 'Ок'.lower()]
    self.aggressive_words = {'verb': set(), 'adj': set(), 'all_words': set(), 'noun': set()}
    self.morph_analyzer = MorphAnalyzer()

    self.stop_words = set(stopwords.words('russian')) # стоп-слова из nltk
    self.stop_words.add('')
    self.stop_words.add(' ')
    self.stop_words.add('\t')

  # возвращает нормальную форму слова(при normal_form=True, иначе просто слово) и его тэг(характеристики слова)
  def word_preprocess(self, word, word_normal_form=False):
      word = re.sub(r'[\d\W]', '', word).lower().strip() # убирает пробелы, цифры и знаки препинания
      word = word.replace('_', '')
      w = self.morph_analyzer.parse(word)[0]
      if word_normal_form:
        return w.normal_form, w.tag
      return word, w.tag

  def train_doc2vec_model(self, X, y, comm_dataset=None):
    print('training vectorizer model...')
    dataset = X.join(y).copy()
    dataset = dataset.fillna({'comment': self.standart_comments[0]})
    data = []
    tag_n = 0
    for row in dataset.itertuples(): # перебираем все строки в датасете
        comment = getattr(row, 'comment')
        comment = [self.word_preprocess(word, word_normal_form=True)[0] for word in comment.split(" ") if self.word_preprocess(word, word_normal_form=True)[0] not in self.stop_words]
        comment = [word for word in comment if word != '']
        if len(comment) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
          data.append(TaggedDocument(comment, [tag_n]))
          tag_n += 1
    if comm_dataset is not None:
      comm_dataset = comm_dataset.fillna({'comment': self.standart_comments[0]})
      for row in comm_dataset.itertuples():
        comment = getattr(row, 'comment')
        comment = [self.word_preprocess(word, word_normal_form=True)[0] for word in comment.split(" ") if self.word_preprocess(word, word_normal_form=True)[0] not in self.stop_words]
        comment = [word for word in comment if word != '']
        if len(comment) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
          data.append(TaggedDocument(comment, [tag_n]))
          tag_n += 1
    self.text_vectorizer.build_vocab(data)
    for epoch in range(30):
      self.text_vectorizer.train(utils.shuffle(data), total_examples=len(data), epochs=1)
      self.text_vectorizer.alpha -= 0.002
      self.text_vectorizer.min_alpha = self.text_vectorizer.alpha
    print('TEXT WECTORIZER TRAINED')
    
  # тренировка векторизатора машин
  def train_cars_vectorizer_and_clusterer(self, X, y):
      print('training vectorizer model...')
      dataset = X.join(y).copy()
      data = []
      tag_n = 0
      for row in dataset.itertuples(): # перебираем все строки в датасете
          if getattr(row, 'mark'):
            car = getattr(row, 'mark')
            car = [word.lower().strip() for word in car.split(" ")]
            car = [word for word in car if word != '']
            if len(car) > 0:
              data.append(TaggedDocument(car, [tag_n]))
              tag_n += 1
      self.cars_vectorizer.build_vocab(data)
      for epoch in range(20):
        self.cars_vectorizer.train(utils.shuffle(data), total_examples=len(data), epochs=1)
        self.cars_vectorizer.alpha -= 0.002
        self.cars_vectorizer.min_alpha = self.cars_vectorizer.alpha
      print('CARS WECTORIZER TRAINED')
      print('training cars clustering model...')
      vectors = []
      c = 0
      for row in dataset.itertuples(): # перебираем все строки в датасете
        if getattr(row, 'mark'):
          car = getattr(row, 'mark')
          car = [word.lower().strip() for word in car.split(" ")]
          car = [word for word in car if word != '']
          if len(car) > 0:
            vectors.append(self.cars_vectorizer.infer_vector(car))
      self.cars_clusterer.fit(vectors)
      print('CARS CLUSTERING COMPLETED')

  def train_comm_model(self, X, y):
    dataset = X.join(y).copy()
    dataset = dataset.fillna({'comment': self.standart_comments[0]})
    vectors = []
    c = 0
    train_y = []
    for row in dataset.itertuples(): # перебираем все строки в датасете
      comment = getattr(row, 'comment')
      comment = [self.word_preprocess(word, word_normal_form=True)[0] for word in comment.split(" ") if self.word_preprocess(word, word_normal_form=True)[0] not in self.stop_words]
      comment = [word for word in comment if word != '']
      if len(comment) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
        vectors.append(self.text_vectorizer.infer_vector(comment))
        train_y.append(getattr(row, 'is_aggressive'))
    self.comm_model.fit(vectors, train_y)
    print('COMM MODEL TRAINED')

  # средний рейтинг по комментариям для каждого водителя  (плохо работает, хз че с ними делать), не юзать пока
  def mean_comments_aggressive_rate(self, comm_dataset, X):
    mean_comments_aggressive_rate = []
    comm_dataset = comm_dataset.fillna({'comment': self.standart_comments[0]})
    for drv_id in X['driver_id']:
      driver_comments_rate = []
      driver_comments = comm_dataset.loc[comm_dataset['driver_id'] == drv_id]
      for row in driver_comments.itertuples():
        commentt = getattr(row, 'comment')
        commentt = [self.word_preprocess(word, word_normal_form=True)[0] for word in commentt.split(" ") if self.word_preprocess(word, word_normal_form=True)[0] not in self.stop_words]
        commentt = [word for word in commentt if word != '']
        if getattr(row, 'comment').lower().strip() not in self.standart_comments and len(commentt) > 0:
          driver_comments_rate.append(self.comm_model.predict_proba([self.text_vectorizer.infer_vector(commentt)])[:, 1][0])
        else:
          driver_comments_rate.append(0)
      if len(driver_comments_rate) == 0:
        driver_comments_rate.append(0)
      mean_comments_aggressive_rate.append(np.mean(driver_comments_rate))
    return mean_comments_aggressive_rate

  # заполнение словаря агрессивными словами
  def fill_agressive_vocab(self, X, y):
    dataset = X.join(y).copy()
    for row in dataset.itertuples(): # перебираем все строки в датасете
      if getattr(row, 'is_aggressive') == 1 and getattr(row, 'comment') and getattr(row, 'comment').lower().strip() not in self.standart_comments:
        words = [self.word_preprocess(word, word_normal_form=True) for word in getattr(row, 'comment').split(' ')]
        for word in words:
          if word[0] not in self.stop_words:
            if 'VERB' in word[1]:
              self.aggressive_words['verb'].add(word[0])
            elif 'ADJF' in word[1] or 'ADJS' in word[1]:
              self.aggressive_words['adj'].add(word[0])
            elif 'NOUN' in word[1]:
              self.aggressive_words['noun'].add(word[0])

  # делаем NLP фичи на основе сгенерированного словаря
  def NLP_feature_extract(self, X, y=None):
    agg_verbs_rate = [] # глаголы
    agg_adjs_rate = [] # прилагательные
    agg_nouns_rate = [] # существительные
    for row in X.itertuples(): # перебираем все строки в датасете
      if getattr(row, 'comment'):
        words = [self.word_preprocess(word, word_normal_form=True) for word in getattr(row, 'comment').split(' ')]
        words_verb = [word[0] for word in words if 'VERB' in word[1] and word[0] not in self.stop_words]
        words_adj = [word[0] for word in words if ('ADJF' in word[1] or 'ADJS' in word[1]) and word[0] not in self.stop_words]
        words_noun = [word[0] for word in words if 'NOUN' in word[1] and word[0] not in self.stop_words]

        if len(words_verb) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
          agg_verbs_rate.append(len(set(words_verb) & self.aggressive_words['verb']) / len(words_verb))
        else:
          agg_verbs_rate.append(0)

        if len(words_adj) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
          agg_adjs_rate.append(len(set(words_adj) & self.aggressive_words['adj']) / len(words_adj))
        else:
          agg_adjs_rate.append(0)

        if len(words_noun) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
          agg_nouns_rate.append(len(set(words_noun) & self.aggressive_words['noun']) / len(words_noun))
        else:
          agg_nouns_rate.append(0)

    return agg_verbs_rate, agg_adjs_rate, agg_nouns_rate

  # отбор фичей
  def features(self, X, comm_dataset=None):
    data = X.copy()
    agg_verbs_rate, agg_adjs_rate, agg_nouns_rate = self.NLP_feature_extract(data)

    data['agg_verbs_rate'] = agg_verbs_rate
    data['agg_adjs_rate'] = agg_adjs_rate
    data['agg_nouns_rate'] = agg_nouns_rate

    feature_list = ['agg_verbs_rate', 'agg_adjs_rate', 'agg_nouns_rate']

    if comm_dataset is not None:
      data['mean_comments_aggressive_rate'] = self.mean_comments_aggressive_rate(comm_dataset, X)
      feature_list.append('mean_comments_aggressive_rate')
    
    probabilities = []
    for row in data.itertuples(): # перебираем все строки в датасете
       comment = getattr(row, 'comment')
       comment = [self.word_preprocess(word, word_normal_form=True)[0] for word in comment.split(" ") if self.word_preprocess(word, word_normal_form=True)[0] not in self.stop_words]
       comment = [word for word in comment if word != '']
       if len(comment) > 0 and getattr(row, 'comment').lower().strip() not in self.standart_comments:
         probabilities.append(self.comm_model.predict_proba([self.text_vectorizer.infer_vector(comment)])[:, 1][0])
       else: 
         probabilities.append(0)

    data['agg_comm_probability'] = probabilities
    feature_list.append('agg_comm_probability')

    cars_cluster = []
    for row in data.itertuples(): # перебираем все строки в датасете
       car = getattr(row, 'mark')
       car = [word.lower().strip() for word in car.split(" ")]
       car = [word for word in car if word != '']
       cars_cluster.append(self.cars_clusterer.predict([self.cars_vectorizer.infer_vector(car)])[0])
    data['cars_cluster'] = cars_cluster
    #feature_list.append('cars_cluster')

    data['is_comment'] = [1 if getattr(row, 'comment') and getattr(row, 'comment').lower().strip() not in self.standart_comments else 0 for row in data.itertuples()]
    feature_list.append('is_comment')

    # заполним NaN средними значениями
    for feature in feature_list:
      data = data.fillna({feature: data[feature].mean()})
    
    data = data.set_index('order_id')
    return data[feature_list]

  # кросс-валидация и предикт на тесте
  def train_eval(self, X, y, comm_dataset_labled=None, comm_dataset_unlabled=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = X_train.fillna({'comment': self.standart_comments[0]})
    X_test = X_test.fillna({'comment': self.standart_comments[0]})

    self.fill_agressive_vocab(X_train, y_train) # заполнение словаря агрессивных слов
    self.train_doc2vec_model(X_train, y_train, comm_dataset_labled) # тренировка doc2vec модели и кластеризации для комментариев о поездке
    self.train_cars_vectorizer_and_clusterer(X_train, y_train) # тренировка doc2vec модели и кластеризации для машин
    self.train_comm_model(X_train, y_train) # тренировка модели вероятностей агрессивности текстов

    X_train_features = self.features(X_train, comm_dataset_labled)

    self.model = LogisticRegression(random_state=42, max_iter=1000)

    cv_score = cross_val_score(self.model, X_train_features, y_train, cv=5, scoring='roc_auc')
    
    self.model.fit(X_train_features, y_train)

    print('Test Roc-Auc score:', roc_auc_score(y_test, self.model.predict_proba(self.features(X_test, comm_dataset_labled))[:, 1]))
    print('Train Roc-Auc score:', roc_auc_score(y_train, self.model.predict_proba(X_train_features)[:, 1]))
    print(f"CV_mean roc_auc: {np.mean(cv_score)}, CV_folds_score: {cv_score}")
    return X_train_features
  
  def predict(self, X, comm_dataset=None):
    res = self.model.predict_proba(self.features(X, comm_dataset))[:, 1]
    X = X.fillna({'comment': self.standart_comments[0]})
    datasss = pd.DataFrame()
    datasss['is_aggressive'] = res
    datasss.to_csv('resss.csv')
    print('prediction saved')

def get_model(X:pd.DataFrame,y:pd.DataFrame, comments:pd.DataFrame):
    model = Model()
    
    features = model.train_eval(X, y, comm_dataset_labled=comments)
    return model,features

