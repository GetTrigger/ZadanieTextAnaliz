# https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe/33912971

import pandas as pd

# Загружаем данные из файлов и объединяем в один дата сет
df_train_statya = pd.DataFrame(pd.read_table('Source_Statya_train.txt', sep='\n', encoding='utf-8', engine='python'))
df_train_statya['Метка'] = 1

df_train_web = pd.DataFrame(pd.read_table('Source_Web_train.txt', sep='\n', encoding='utf-8', engine='python'))
df_train_web['Метка'] = 0

df_test_statya = pd.DataFrame(pd.read_table('Source_Statya_test.txt', sep='\n', encoding='utf-8', engine='python'))
df_test_statya['Метка'] = 1

df_test_web = pd.DataFrame(pd.read_table('Source_Web_test.txt', sep='\n', encoding='utf-8', engine='python'))
df_test_web['Метка'] = 0

df = pd.concat([df_train_statya, df_train_web, df_test_statya, df_test_web], ignore_index=True)

# Проверка на отсутствие пустых полей
df['Литературные источники'].isnull().sum()
print("Количество Статей из Журналов: ", df[df['Метка'] == 1].shape[0])
print("Количество Статей из Веб ресурса: ", df[df['Метка'] == 0].shape[0])

# Работа с данными
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('russian')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def data_preprocessing(full_data):
    # Очистка Данных
    # Уменьшение регистра
    full_data = full_data.lower()

    # Токенизация
    tokens = nltk.word_tokenize(full_data)

    # Удаление Стоп-слов
    full_data = [word for word in tokens if word not in stop_words]

    # Стемминг
    # full_data = [stemmer.stem(word) for word in full_data]

    # Лемматизация
    # full_data = [lemmatizer.lemmatize(word) for word in full_data]

    full_data = ' '.join(full_data)

    return full_data


# Отображение данных после очистки
df['Редактированное'] = df['Литературные источники'].apply(lambda review: data_preprocessing(review))
print(df.head(10))

# Разделение данных 70 на 30 для Обучения и Тестирования
from sklearn.model_selection import train_test_split

data = df.copy()
y = data['Метка'].values
data.drop(['Метка'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, stratify=y)

print('Данные для Обучения:', X_train.shape, y_train.shape)
print('Данные для Проверки:', X_test.shape, y_test.shape)

# Векторизация текстовых данных
# BOW

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=5)

X_train_full_data_bow = vect.fit_transform(X_train['Редактированное'])
X_test_full_data_bow = vect.transform(X_test['Редактированное'])

print('BoW')
print('Данные для Обучения: ', X_train_full_data_bow.shape)
print('Данные для Проверки: ', X_test_full_data_bow.shape)

# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5)

X_train_full_data_tfidf = vectorizer.fit_transform(X_train['Редактированное'])
X_test_full_data_tfidf = vectorizer.transform(X_test['Редактированное'])

print('TF-IDF')
print('Данные для Обучения: ', X_train_full_data_tfidf.shape)
print('Данные для Проверки: ', X_test_full_data_tfidf.shape)

# Создание моделей
# Наивный Байесовский: BoW'
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB()
clf.fit(X_train_full_data_bow, y_train)

y_pred = clf.predict(X_test_full_data_bow)
print('Наивный Байесовский: BoW')
print('Точность: ', accuracy_score(y_test, y_pred))
x1 = y_pred

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Матрица неточностей')
plt.show()

# Наивный Байесовский: TF-IDF
clf = MultinomialNB(alpha=1)
clf.fit(X_train_full_data_tfidf, y_train)

y_pred = clf.predict(X_test_full_data_tfidf)
print('Наивный Байесовский: TF-IDF')
print('Точность: ', accuracy_score(y_test, y_pred))
x2 = y_pred

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Матрица неточностей')
plt.show()

# Логистическая Регрессия: BoW
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', solver='liblinear')

clf.fit(X_train_full_data_bow, y_train)

y_pred = clf.predict(X_test_full_data_bow)
print('Логистическая Регрессия: BoW')
print('Точность: ', accuracy_score(y_test, y_pred))
x3 = y_pred

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Матрица неточностей')
plt.show()

# Логистическая Регрессия: TFIDF
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', solver='liblinear')

clf.fit(X_train_full_data_tfidf, y_train)

y_pred = clf.predict(X_test_full_data_tfidf)
print('Логистическая Регрессия: TFIDF')
print('Точность: ', accuracy_score(y_test, y_pred))
x4 = y_pred

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Матрица неточностей')
plt.show()

# Сравнение эффективности
print('Сравнение эффективности')
from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ['Векторизация', 'Модель', 'Точность']
x.add_row(['BOW', 'Наивный Байесовский', accuracy_score(y_test, x1)])
x.add_row(['TFIDF', 'Наивный Байесовский', accuracy_score(y_test, x2)])
x.add_row(['BOW', 'Логистическая Регрессия', accuracy_score(y_test, x3)])
x.add_row(['TFIDF', 'Логистическая Регрессия', accuracy_score(y_test, x4)])
print(x)
