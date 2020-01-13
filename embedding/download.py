import re
import urllib2
import csv
import contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from bs4 import BeautifulSoup
import os.path

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def replace_contractions_numbers(text):
    re.sub('\d', 'number', text)
    return contractions.fix(text)

def sent_tokenize(text):
    sentences =re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",text)
    sentences = [re.sub('\W+', ' ',x) for x in sentences]
    sentences=[x.strip() for x in sentences]
    return ''.join(sentences)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = replace_contractions_numbers(text)
    text = sent_tokenize(text)
    text = text.translate(string.punctuation)
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def downloadFile(url,counter):
    try:
        request = urllib2.Request(url)
        f = urllib2.urlopen(request, timeout=4).read()
        filename = ""+str(counter)+ ".txt"
        # if os.path.isfile("files/"+filename):
        #     print "Exists"
        # else:
        # file = open("files/"+filename, "w")
        html = denoise_text(f).encode('utf8')
        # file.write(html)
        # file.close()
        return html
    except Exception as e:
        print url
        print(e)


# dataset = open('queue.txt').readlines()
progress=0
with open('datasetGenerated.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    with open('data_file_Generated.csv', mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            print (progress)
            data_writer.writerow([downloadFile(row[0],progress),row[1]])
            progress = progress + 1

#
# progress=0
# for link in dataset:
#     print (progress)
#     progress= progress + 1
#     link=downloadFile(link)
# dataset = np.asarray(dataset)
# print(dataset.shape)