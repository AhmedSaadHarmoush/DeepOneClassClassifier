import re

import pandas as pd
import numpy as np
import pickle
from urlparse import urlparse
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import warnings
import csv
import sys
csv.field_size_limit(sys.maxsize)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


warnings.filterwarnings('ignore')


# sns.set(style='whitegrid', palette='muted', font_scale=1.5)
# rcParams['figure.figsize'] = 14, 8
LABELS = [-1, 1]

# newsDS = pd.read_json("News_Category_Dataset_v2.json" , lines='true')
# # print newsDS.columns
#
# newsDS['full'] = newsDS['headline'] +  newsDS['short_description']
# newsDS['y']=1
# newsDS.drop(newsDS[newsDS['short_description'] == '' ].index, inplace=True)
# for i in newsDS[newsDS['category'] == 'POLITICS' ].index:
#     newsDS['y'][i]=-1
# print newsDS.head()


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
            print progress
            if progress>4000:
                break;
#print np.shape(newsDS)
X_train, X_test, y_train, y_test = train_test_split(newsDS, y,  test_size = 0.1)
newDs=None
y=None
# X_train = X_train[X_train['y'] == 1]
#
# y_train = X_train['y']
# y_test = X_test['y']
#
# X_train = X_train['full']
# X_test = X_test['full']

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)

X_train.shape

train_set_len = len(X_train)
test_set_len = len(X_test)
validate_set_len = len(X_val)

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
def process_sentences(main_text):
    headlines_without_numbers = re.sub('[^a-zA-Z]', ' ', main_text)
    words = word_tokenize(headlines_without_numbers.lower())
    stop_words_english = set(stopwords.words('english'))
    final_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english]
    return(' '.join(final_words))


finalHeadlineTrain = []
finalHeadlineTest = []
finalHeadlineVal = []
fullText=[]
for i in range(0, train_set_len):
    finalHeadlineTrain.append(process_sentences(X_train[i]))
    fullText.append(process_sentences(X_train[i]))
for i in range(0, validate_set_len):
    finalHeadlineVal.append(process_sentences(X_val[i]))
    fullText.append(process_sentences(X_val[i]))
for i in range(0, test_set_len):
    finalHeadlineTest.append(process_sentences(X_test[i]))
    fullText.append(process_sentences(X_test[i]))
X_train=None
X_test=None
X_val=None

vectorizerTf = TfidfVectorizer()
vectorizerTf.fit(finalHeadlineTrain)
bagOfWords_train_tf = vectorizerTf.transform(finalHeadlineTrain)
tfidX_train = bagOfWords_train_tf.toarray()
bagOfWords_train_tf=None
finalHeadlineTrain=None
bagOfWords_val_tf = vectorizerTf.transform(finalHeadlineVal)
tfidX_val = bagOfWords_val_tf.toarray()
bagOfWords_val_tf=None
finalHeadlineVal=None
bagOfWords_test_tf = vectorizerTf.transform(finalHeadlineTest)
tfidX_test = bagOfWords_test_tf.toarray()
bagOfWords_test_tf=None
finalHeadlineTest=None
print "train"
print np.shape(tfidX_train)
print "test"
print np.shape(tfidX_test)
print "Validate"
print np.shape(tfidX_val)

print "---------------------------------------------"
print "TFIDF Building :"


# threshold=0.5
error_df = pd.DataFrame({'reconstruction_error': y_test,
                        'true_class': y_test})
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
threshold=np.percentile(y_test,80)
# for name, group in groups:
#     ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
#             label= "1" if name == 1 else "-1")
# ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("classes before classification")
# plt.ylabel("Reconstruction error")
# plt.xlabel("Data point index")
# plt.show();



input_dim = 1000
encoding_dim = 700

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 10
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(tfidX_train, tfidX_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(tfidX_val, tfidX_val),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history



plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


predictions = autoencoder.predict(tfidX_test)
mse = np.mean(np.power(tfidX_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
print error_df.describe()



fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()


plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()


threshold = np.percentile(mse,80)

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "1" if name == 1 else "-1")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


y_pred = [1 if e > threshold else -1 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

print "TFIDF - Autoencoder :"
target_names = ['class -1', 'class 1']
print np.shape(y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))

