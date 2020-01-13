# DeepOneClassClassifier
Open source implementation for one-class classification using multiple models including an autoencoder
That goes through multiple steps as follows

## Word Embedding
- using Countizer
- Tf-Idf
- Doc2Vec
- fastText

## classification
- LogisticRegression
- Linear Kernal SVM
- Non-linear Kernal SVM
- One-class SVM
- Autoencoder

## How to use
- You need to have python2.7
- git clone https://github.com/AhmedSaadHarmoush/DeepOneClassClassifier.git
- feed data to model 

## Sample Output
### tf-idf
  1.90888405e+00 -1.62066057e-01 -5.36465108e-01 -9.49786544e-01
 -4.38950211e-01 -1.31349492e+00 -2.26326242e-01 -5.51806800e-02
 -1.41368866e+00  4.22046870e-01 -1.49572647e+00 -1.61181796e+00
  6.15626037e-01  1.04208028e+00  8.66693616e-01  1.16182303e+00
 -7.42550254e-01  3.09558325e-02  1.48470029e-01  3.12349461e-02
  5.50389230e-01  7.11790845e-02 -7.69299150e-01 -9.34180915e-01
 -5.02596378e-01 -1.59273672e+00 -2.49899209e-01 -1.40247250e+00
 ### TFIDF - SVM :
              precision    recall  f1-score   support

    class -1       0.78      0.88      0.82         8
     class 1       0.60      0.50      0.55         6
     class 2       0.78      0.78      0.78         9
     class 3       0.69      1.00      0.82         9
     class 4       1.00      0.50      0.67         8

   micro avg       0.75      0.75      0.75        40
   macro avg       0.77      0.73      0.73        40
weighted avg       0.78      0.75      0.74        40

 
