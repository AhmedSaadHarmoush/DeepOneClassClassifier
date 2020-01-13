import timeit
import csv
import sys
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
csv.field_size_limit(sys.maxsize)


class Embedding (object):
    if __name__ == '__main__':
        newsDS = {'data':[],'y':[]}
        y = []
        df = pd.read_csv('../data/data_file_Generated.csv',nrows=5000)
        print df.head()
        print df['data'].shape[0]
        #
        sentences_train, sentences_test = train_test_split(df, test_size=0.25,random_state=1000)
        #Fast Text
        # start = timeit.default_timer()
        # vectorizer = FastText(df['data'],size=1000, iter=30)
        # vectorizer.build_vocab(df['data'],update=True)
        # print "Build Vocab Time"
        # print timeit.default_timer() - start
        # start = timeit.default_timer()
        # vectorizer.train(total_examples=df['data'].shape[0],sentences=sentences_train['data'], epochs=vectorizer.epochs)
        # print vectorizer.wv[df['data'][0]]
        # print "Train Time"
        # print timeit.default_timer() - start
        # TfIdf
        # vectorizer = TfidfVectorizer(max_features=1000)
        # vectorizer.fit(sentences_train['data'])
        #
        # X_train = vectorizer.transform(sentences_train['data'])
        # y_train = sentences_train['y']
        #
        # sentences_train_true = sentences_train[sentences_train['y']==1]
        # X_train_true = vectorizer.transform(sentences_train_true['data'])
        #
        # X_test = vectorizer.transform(sentences_test['data'])
        # y_test = sentences_test['y']
        #
        # sentences_test_true = sentences_test[sentences_test['y'] == 1]
        # X_test_true = vectorizer.transform(sentences_test_true['data'])
        # print np.shape(df)
        # print np.shape(X_train)
        # print np.shape(X_train_true)
        # print np.shape(X_test)
        # print np.shape(X_test_true)
        # #
        start = timeit.default_timer()
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(df['data'])]
        tagged_data_train = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentences_train['data'])]
        modelDoc2vec = Doc2Vec(vector_size=1000, min_count=1, dm=1)
        modelDoc2vec.build_vocab(tagged_data)
        print "Build Vocab Time"
        print timeit.default_timer() - start
        for epoch in range(30):
            print('iteration {0}'.format(epoch))
            modelDoc2vec.train(tagged_data, total_examples=modelDoc2vec.corpus_count, epochs=modelDoc2vec.iter)
            modelDoc2vec.alpha -= 0.0002
            modelDoc2vec.min_alpha = modelDoc2vec.alpha
        print "Train Time"
        print timeit.default_timer() - start
        # print tagged_data
        # str = " "
        # denseX = modelDoc2vec.docvecs[0]
        # print denseX
"""
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print "---------------------------------------------"
        print "TFIDF - LogisticRegression :"
        print("Accuracy:", score)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))




        svm_model = svm.SVC(kernel='sigmoid' ,gamma='auto')
        svm_model.fit(X_train,y_train)
        predictions = svm_model.predict(X_test)
        print "---------------------------------------------"
        print "TFIDF - SVM Sigmoid:"
        score = svm_model.score(X_test, y_test)
        print("Accuracy:", score)
        print(classification_report(y_test, predictions))


        svm_model = svm.SVC(kernel='linear' ,gamma='auto')
        svm_model.fit(X_train,y_train)
        predictions = svm_model.predict(X_test)
        print "---------------------------------------------"
        print "TFIDF - SVM  linear:"
        score = svm_model.score(X_test, y_test)
        print("Accuracy:", score)
        print(classification_report(y_test, predictions))

        svm_model = svm.OneClassSVM( kernel="sigmoid" ,gamma='auto')
        svm_model.fit(X_train,y_train)
        predictions = svm_model.predict(X_test)
        print "---------------------------------------------"
        print "TFIDF - OneClassSVM Sigmoid:"
        print(classification_report(y_test, predictions))

        svm_model = svm.OneClassSVM(gamma='auto' , kernel="rbf")
        svm_model.fit(X_train,y_train)
        predictions = svm_model.predict(X_test)
        print "---------------------------------------------"
        print "TFIDF - OneClassSVM rbf :"
        print(classification_report(y_test, predictions))

        svm_model = svm.OneClassSVM(gamma='auto' , kernel="linear")
        svm_model.fit(X_train,y_train)
        predictions = svm_model.predict(X_test)
        print "---------------------------------------------"
        print "TFIDF - OneClassSVM linear :"
        print(classification_report(y_test, predictions)) """
