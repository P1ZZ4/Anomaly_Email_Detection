# 해당 코드는 jupyter lab에서 돌린 code이며, 편의상 py로 올려놓음 

from sklearn.metrics import accuracy_score
import csv
import nltk
import re
from sklearn.cluster import KMeans
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer

csv.field_size_limit(100000000)

import sys
import csv

def reading_training_file():
    documents = []
    with open("1113_category.csv","rt") as csvfile:
        rows=csv.reader(csvfile)
        next(rows, None) 
        for row in rows:
            message = row[1]
            documents.append(message) 
    return documents
  
def preprocessing(documents):
    vocab_glob = {}
    tokenized_document = []
    final_documents=[]
    for document in documents:
        text=document.replace("</p>","") 
        text=text.replace("<p>"," ") 
        text = text.replace("http", " ")
        text = text.replace("www", " ")
        text = re.sub(r'([a-z])\1+', r'\1', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('\.+', '.', text)
        text = re.sub(r"(?:\@|'|https?\://)\s+","",text) 
        text = re.sub("[^a-zA-Z]", " ",text)
        text=re.sub(r'[^\w\s]','',text) 
        text=re.sub("\d+","",text) 
        tokens_text = nltk.word_tokenize(text) 
        stopwords=nltk.corpus.stopwords.words('english') 
        tokens_text=[w for w in tokens_text if w.lower() not in stopwords]
        tokens_text=[w.lower() for w in tokens_text] 
        tokens_text=[w for w in tokens_text if len(w)>2] 
        p= PorterStemmer() 
        tokens_text = [p.stem(w) for w in tokens_text]
        token_ind= []
        counter=len(vocab_glob)-1
        for token in tokens_text:
         if token not in vocab_glob:
            counter+=1
            vocab_glob[token]=counter
            token_ind.append(counter)
         else:
            token_ind.append(vocab_glob[token])
        final_documents.append(token_ind)
    return vocab_glob,final_documents
  
def feature_selection(final_documents):
 doc_freq = {}
 for document in final_documents:
   for index in document:
    if index in doc_freq:
        doc_freq[index] += 1
    else:
        doc_freq[index] = 1
    top_features = []
    for token in doc_freq.keys():
     if doc_freq[token] >10:    
      top_features.append(token)
 i = 0
 top_words = {}
 for token in top_features:
        top_words[i] = token
        i += 1
 return top_words

def features_vector_binary(vocab_glob,final_documents):
    indexes_features=vocab_glob.values()
    rows=[]
    rows=indexes_features
    columns=[]
    values=[]

    for val in final_documents:
        feature_vector = [0]*(len(indexes_features))
        for j in val:
         if j in rows:
             counter=1
             feature_vector[rows.index(j)] = counter
        columns.append(feature_vector)
    values=np.array(columns)
    return values

def feature_matrix_term_frequency(top_words, final_documents):
 print (final_documents)
 indexes_features = top_words.values()
 print (indexes_features)

 rows = []
 rows = indexes_features
 columns = []
 values = []

 for val in final_documents:
     feature_vector = [0] * (len(indexes_features))
     for j in val:
         if j in rows:
             counter = 1
             feature_vector[rows.index(j)] = val.count(j)
     columns.append(feature_vector)
 values = np.array(columns)
 print (values.shape)
 return values

def feature_matrix_tfidf(top_words,final_documents):
    print ("Calculating tfidf weight using tfidf transformer...")
    #print final_documents
    indexes_features = top_words.values()
    rows = []
    rows = indexes_features
    columns = []
    values = []

    for val in final_documents:
        feature_vector = [0] * (len(indexes_features))
        for j in val:
            if j in rows:
                feature_vector[rows.index(j)] = val.count(j)
        columns.append(feature_vector)


    tfidf = TfidfTransformer(norm=False,use_idf=True,sublinear_tf=True, smooth_idf=True)
    tfidf.fit(columns)
    tfidf_matrix = tfidf.transform(columns)
    test=tfidf_matrix
    return test
  
def kmeans(f_vector):
    num_clusters = 2
    km = KMeans(num_clusters,random_state=99,init='k-means++', n_init=14, max_iter=100, tol=0.00001, copy_x=True)
    km.fit(f_vector)
    clusters = km.labels_.tolist()
    print ("Results of Clustering:")
    print (clusters)
    print ("Length of results:")
    print (len(clusters))
    return clusters
  
def write_to_csv(clusters):
   file_save = "final_results.csv"
   with open(file_save, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["SMS_id", "label"])
    numbers = list(range(1, len(clusters)+1))
    for row in zip(numbers, clusters):
     writer.writerow(row)


   f_save = "clustering_results.csv"
   with open(f_save, 'wb') as f:
    writer = csv.writer(f)
    for row in clusters:
        writer.writerow([row])
        
 list_prediction=[]
    list_true = []
    with open("clustering_results.csv","rt") as f_pre:
        rows= csv.reader(f_pre)
        for row in rows:
            if row[0]==str(0):
                list_prediction.append(int(row[0]))
            elif row[0]==str(1):
                list_prediction.append(int(row[0]))

    with open("accurate_results.csv","rt") as f:
        rows= csv.reader(f)
        for row in rows:
            if row[0]==str(0):
                list_true.append(0)
            elif row[0]==str(1):
                list_true.append(1)
    return list_prediction, list_true
  
  
def main():
    documents = reading_training_file()
    vocab_glob,final_documents=preprocessing(documents)
    top_words= feature_selection(final_documents)
    feature_vector=feature_matrix_tfidf(top_words, final_documents)
    clusters=kmeans(feature_vector)
    write_to_csv(clusters)
    y_pre, list_true = read_accuracy_file()
    accuracy =  accuracy_score(list_true, y_pre)
    print ("Accuracy:", accuracy)
main()
