# -*- coding:utf-8 -*-

import os
import tarfile
import numpy as np
import re
import chardet
import nltk
import scipy
from gensim.models import word2vec, Word2Vec
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import io
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC


class Kernel(object):
    def __init__(self, type='linear', p=None):
        self.p = p
        self.type = type

    def _linear_kernel_(self, U, V):
        return np.matmul(U, np.transpose(V))

    def _poly_kernel_(self, U, V, p):
        return np.exp(np.add(1, np.matmul(U, np.transpose(V))), p)

    def _rbf_kernel_(self, U, V, p):
        m = np.shape(U)[0]
        n = np.shape(V)[0]
        U2 = np.tile(np.sum(U * U, 1), [1, n])
        V2 = np.tile(np.sum(V * V, 0), [m, 1])
        UV = 2 * np.dot(U, np.transpose(V))
        return np.exp(np.divide(U2 + V2 - 2 * UV, 2 * np.power(p, 2)))

    def compute(self, U, V):
        if self.type == 'linear':
            self._linear_kernel_(U, V)
        elif self.type == 'poly':
            self._poly_kernel_(U, V, self.p),
        elif self.type == 'rbf':
            self._rbf_kernel_(U, V, self.p)
        else:
            raise AssertionError()


class MTPSVM(object):

    def __init__(self, c, mu, kernel):
        self.c = c
        self.mu = mu
        self.kernel = kernel

    def fit(self, X, y):
        num_tasks = X.shape[0]
        X = np.array(X)
        y = np.array(y)
        X = [item for task in X for item in task]
        sample_per_task = [task.shape[0] for task in X]
        K = self.kernel.compute(X, X)
        Q = np.multiply(np.multiply(X, y), y.T)
        delta = np.repeat(np.array(range(0, num_tasks)), sample_per_task)
        for i in delta:
            for j in delta:
                if i == j:
                    Q[i, j] = Q[i, j] * (1 + num_tasks / self.mu + 1)
        alpha = scipy.invert(Q) * np.ones(y.shape[0])
        pass

    def predict(self):
        pass


def processing(text):
    text = re.sub('\d+', '', text)
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    # 去除停用词
    stops = stopwords.words('english')
    tokens = [token for token in tokens if tokens not in stops]
    # 去除长度小于3的词
    tokens = [token for token in tokens if len(token) >= 3]
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def load_files(folder):
    files = os.listdir(folder)
    contents = []
    for file in files:
        try:
            lines = open(folder + file).readlines()
            contents.append(processing(''.join(lines)))
        except UnicodeDecodeError:
            print('exception in:%s' % file)
            f = open(folder + file, 'rb')
            char = chardet.detect(f.read())
            print(char)
            lines = open(folder + file, encoding=char['encoding'], errors='ignore').readlines()
            contents.append(processing(''.join(lines)))
        finally:
            pass
    return contents


path = './Spambase/enron/'
if not os.path.exists(path):
    try:
        os.mkdir(path)
        for i in range(6):
            filename = './Spambase/enron{1}.tar.gz'
            tar = tarfile.open(filename.format(i + 1))
            tar.extractall(path=path)
            tar.close()
            print('extract:%s' % filename)
    except:
        os.remove(path)
    finally:
        print('extract finished')
else:
    if os.path.exists('tasks.npz'):
        print('file existed')
        tasks = np.load('tasks.npz')['str']
    else:
        ham_folder = path + 'enron%d/ham/'
        spam_folder = path + 'enron%d/spam/'
        # 得到所有任务的原始数据
        tasks = []
        for i in range(6):
            ham_data = load_files(ham_folder % (i + 1))
            spam_data = load_files(spam_folder % (i + 1))
            data = ham_data + spam_data
            label = np.append(np.zeros(len(ham_data)), np.ones(len(spam_data)))
            print('task:%d,ham:%d,spam:%d' % (i + 1, len(ham_data), len(spam_data)))
            tasks.append([data, label])
        np.savez('tasks.npz', str=tasks)

    # 将原始数据转换为Word2Vec表示
    # if os.path.exists(''):
    #     print('load data-word2vec.npy')
    #     [X, Y] = np.load('data.npy')
    #     # feature_names = np.load('features.npy')
    #     # print(X.shape)
    #     # print(feature_names)
    # else:
    #     print('transform word2vec')
    #     raw_documents = [data for task in tasks for data in task[0]]
    #     raw_labels = [label for task in tasks for label in task[1]]
    #     model = word2vec.Word2Vec()
    #     model.train(raw_documents)
    #     req_count = 5
    #     for key in model.wv.similar_by_word('happy', topn=100):
    #         if len(key[0]) == 3:
    #             req_count -= 1
    #             print(key[0], key[1])
    #             if req_count == 0:
    #                 break

    # 将原始数据转换为TF-IDF表示
    if os.path.exists('data.npy'):
        print('load data.npy')
        [X, Y] = np.load('data.npy')
        feature_names = np.load('features.npy')
        print(X.shape)
        print(feature_names)
    else:
        print('transform tf-idf')
        raw_documents = [data for task in tasks for data in task[0]]
        raw_labels = [label for task in tasks for label in task[1]]
        vectorizer = CountVectorizer()
        transformer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
        X = transformer.fit_transform(raw_documents)
        Y = raw_labels
        print(X.shape)
        np.save('data', [X, Y])
        np.save('features', transformer.get_feature_names())

    # 保存Matlab格式
    if os.path.exists('data.mat'):
        print('load data.bat')
        data = io.loadmat('data.mat')
    else:
        print('save data.bat')
        io.savemat('data', {'X': X, 'Y': Y})

    # 采用字典学习进行降维
    # if os.path.exists('data_sparse.mat'):
    #     print('data-sparse.mat existed')
    #     X_new = io.loadmat('data-sparse.mat')
    # else:
    #     print('reduce dimensions')
    #     dl = DictionaryLearning(100)
    #     X_new = dl.fit_transform(np.array(X))
    #     print('save data-sparse.mat')
    #     io.savemat('data-sparse', {'X': X_new, 'Y': Y})

    # t = TSNE.fit_transform(X_train, Y_train)
    # PCA.fit_transform()
    # 朴素贝叶斯分类器
    # clf = MultinomialNB()
    # clf.fit(X_train, Y_train)
    # y_nb_pred = clf.predict(X_train)
    # print(y_nb_pred)
    # print(np.mean(y_nb_pred == Y_train)),33716
    #
    # 交叉验证模型
    clf = LinearSVC()
    scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    print(scores)

    # sections =
    # X = np.split(X, sections)
    y = np.split(Y, np.cumsum([5172, 5857, 5512, 6000]))
    clf = MTPSVM(c=1, mu=1, kernel=Kernel())
    clf.fit(X, y)

    # clf.fit(X_train[:1000], Y_train[:1000])
    # y_svm_pred = clf.predict(X_train[1000:])
    # print('svm_confusion_matrix:')
    # cm = confusion_matrix(Y_train[1000:], y_svm_pred)
    # print(cm)
    # print('svm_classification_report:')
    # print(classification_report(Y_train[1000], y_svm_pred))
