import numpy as np
import pandas as pd
# import torch
from sklearn.model_selection import train_test_split    
# tes train split to split test and train data of a dataset
from sklearn.linear_model import LogisticRegression
# Logistic regression is a machine learning algorithm that predicts the probability of certain classes based on dependent variables. It's a supervised learning technique that uses a given set of independent variables to predict the output of a categorical dependent variable
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# TF-IDF (Term Frequency-Inverse Document Frequency) is a text vectorizer that transforms text into a usable vector. It combines two concepts: 
# Term Frequency (TF): The number of times a specific term appears in a document
# Document Frequency (DF): The number of occurrences of a term in a document set

# prepocessing 
data = pd.read_csv('mail_data.csv')
# print(data)

newdata = data.where((pd.notnull(data)),'')

print(newdata.shape)

# label encoding 
# spam = 0
# ham = 1 

newdata.loc[newdata['Category']== 'spam','Category'] = 0 
newdata.loc[newdata['Category']== 'ham','Category'] = 1

# can be used to clean a file 
import string
import re
def clean_text(s): 
    for cs in s:
        if  not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')
def remove_little(s): 
    wordsList = s.split()
    k_length=2
    resultList = [element for element in wordsList if len(element) > k_length]
    resultString = ' '.join(resultList)
    return resultString

newdata = newdata.applymap(lambda x: x.lower() if isinstance(x, str) else x)

newdata['Message'] = newdata['Message'].apply(lambda x: clean_text(x))
newdata['Message'] = newdata['Message'].apply(lambda x: remove_little(x))


print(newdata)

label_counts = data['Category'].value_counts()
percentage_spam = (label_counts[0] / len(data)) * 100
percentage_ham = (label_counts[1] / len(data)) * 100

print(f"Percentage of spam emails: {percentage_spam:.2f}%")
print(f"Percentage of ham emails: {percentage_ham:.2f}%")


X = newdata['Message']
y = newdata['Category']
# print(X,y)

# train_test_split
Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size=0.2,random_state=42)

print(len(Xtrain)+len(ytest)==len(X))

# feature extraction


# Feature extraction using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
Xtest_tfidf = tfidf_vectorizer.transform(Xtest)

ytrain = ytrain.astype('int')
ytest = ytest.astype('int')


def naivebyers():
    model1 = MultinomialNB()
    model1.fit(Xtrain_tfidf, ytrain)

    # Make predictions on the test data
    y_pred_on_train_data = model1.predict(Xtrain_tfidf)  
    accuracy_on_training_data = accuracy_score(ytrain,y_pred_on_train_data)


    ypred_on_test_data = model1.predict(Xtest_tfidf)
    accuracy_on_test_data = accuracy_score(ytest,ypred_on_test_data)


    print(f'Accuracy on training data : {accuracy_on_training_data}')
    print(f'Accuracy on test data : {accuracy_on_test_data}')

    print(len(ytest),len(ypred_on_test_data))

    confusion_mat = confusion_matrix(ytest, ypred_on_test_data)
    report = classification_report(ytest, ypred_on_test_data)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", report)

    return model



def logiscticreg():
    model = LogisticRegression()
    model.fit(Xtrain_tfidf,ytrain)
    
    # evaluate
    y_pred_on_train_data = model.predict(Xtrain_tfidf)  
    accuracy_on_training_data = accuracy_score(ytrain,y_pred_on_train_data)


    ypred_on_test_data = model.predict(Xtest_tfidf)
    accuracy_on_test_data = accuracy_score(ytest,ypred_on_test_data)


    print(f'Accuracy on training data : {accuracy_on_training_data}')
    print(f'Accuracy on test data : {accuracy_on_test_data}')

    print(len(ytest),len(ypred_on_test_data))

    confusion_mat = confusion_matrix(ytest, ypred_on_test_data)
    report = classification_report(ytest, ypred_on_test_data)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", report)

    return model

# print("logistic regression \n")
# logiscticreg()
# print("Naive byers \n")
# naivebyers()


a = ['''User Urgent, Congratulations on winning 20,000 pounds, give us the details of the credit card to proceed and claim your reward instantly''']

a_f = tfidf_vectorizer.transform(a)
print(a_f)


model = logiscticreg()
output = model.predict(a_f)
if output == 1 : print("ham") 
else: print("spam")

model = naivebyers()
output = model.predict(a_f)
if output == 1 : print("ham") 
else: print("spam")
