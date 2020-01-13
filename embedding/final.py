import csv
import sys
import os
import numpy as np
from keras.layers import Dense, regularizers, Embedding, LSTM
from sklearn import svm

from Input.sequential import top_words

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import classification_report

csv.field_size_limit(sys.maxsize)
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

newsDS =[]
y=[]
with open('data_file_Generated.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    progress = 0;
    for row in csv_reader:
        if row[0]:
            newsDS.append(row[0])
            y.append(int(row[1]))
            progress=progress+1
            if progress>=5000:
                break;
sentences_train, sentences_test, y_train, y_test = train_test_split(newsDS, y, test_size=0.25, random_state=1000 )

vectorizer = CountVectorizer(max_features=1000)
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print "---------------------------------------------"
# print "TFIDF - LogisticRegression :"
# print("Accuracy:", score)
# y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
#
#
#
#
# svm_model = svm.SVC(kernel='sigmoid' ,gamma='auto')
# svm_model.fit(X_train,y_train)
# predictions = svm_model.predict(X_test)
# print "---------------------------------------------"
# print "TFIDF - SVM :"
# # print(y_test)
# # print(predictions)
# score = svm_model.score(X_test, y_test)
# print("Accuracy:", score)
# print(classification_report(y_test, predictions))
#
# svm_model = svm.SVC(kernel='linear' ,gamma='auto')
# svm_model.fit(X_train,y_train)
# predictions = svm_model.predict(X_test)
# print "---------------------------------------------"
# print "TFIDF - SVM  linear:"
# # print(y_test)
# # print(predictions)
# score = svm_model.score(X_test, y_test)
# print("Accuracy:", score)
# print(classification_report(y_test, predictions))
#
# svm_model = svm.OneClassSVM( kernel="sigmoid" ,gamma='auto')
# svm_model.fit(X_train,y_train)
# predictions = svm_model.predict(X_test)
# # predictions = [float(x) for x in predictions]
# print "---------------------------------------------"
# print "TFIDF - OneClassSVM :"
#
# print(classification_report(y_test, predictions))
#
# svm_model = svm.OneClassSVM( gamma='auto')
# svm_model.fit(X_train,y_train)
# predictions = svm_model.predict(X_test)
# # predictions = [float(x) for x in predictions]
# print "---------------------------------------------"
# print "TFIDF - OneClassSVM rbf :"
# score = svm_model.score_samples(X_test)
# print("Accuracy:", score)
# print(classification_report(y_test, predictions))
#

from keras.models import Sequential
from keras import layers, Input, Model

input_length = X_train.shape[1]   # Number of features
input_dim = X_train.shape[0]
# for i in range(0,input_dim):
#     X_train[i] = X_train[i] * y_train[i]
# for i in range(0, X_test.shape[0]):
#     X_test[i]=X_test[i] * y_test[i]
# model = Sequential()
embedding_dim = input_length / 2
# model.add(layers.Embedding(input_dim=input_dim,
#                            output_dim=embedding_dim,
#                            input_length=input_length))
# model.add(layers.GlobalMaxPool1D())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()
# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_length, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Embedding(top_words, embedding_dim, input_length=input_length))
model.add(LSTM(100))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# input_layer = Input(shape=(input_length, ))
# encoder = Dense(embedding_dim, activation="tanh",
#                 activity_regularizer=regularizers.l1( 1e-7))(input_layer)
# encoder = Dense(int(embedding_dim/2 ), activation="relu")(encoder)
# decoder = Dense(int(embedding_dim /2), activation='tanh')(encoder)
# decoder = Dense(input_length, activation='relu')(decoder)
# model = Model(inputs=input_layer, outputs=decoder)
#
model.compile(loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy'])
print model.summary()
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10).history
print "---------------------------------------------"
print "TFIDF - Autoencoder :"
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
print ("Loss : {:.4f}".format(loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print ("Loss : {:.4f}".format(loss))

x_pred = model.predict(X_test)
mse = np.mean(np.power(X_test - x_pred, 2), axis=1)
threshold = np.percentile(mse ,30)
y_pred = [1 if e > threshold else -1 for e in mse]
print(classification_report(y_test, y_pred))



