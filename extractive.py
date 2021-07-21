import re
import os
import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import numpy as np
import nltk
import sys

def get_sentences(article):
    extracts=sent_tokenize(article)
    sentences=[]
    for extract in extracts:
        #print(extract)
        clean_sentence=extract.replace("[^a-zA-Z0-9]"," ")   ## Removing special characters
        #print(clean_sentence)
        obtained=word_tokenize(clean_sentence) 
        #print(obtained)
        sentences.append(obtained)

    return sentences

def get_similarity(sent_1,sent_2,stop_words):
  
    sent_1=[w.lower() for w in sent_1]
    sent_2=[w.lower() for w in sent_2]

    total=list(set(sent_1+sent_2)) ## Removing duplicate words in total set

    vec_1= [0] * len(total)
    vec_2= [0] * len(total)


    ## Count Vectorization of two sentences
    for w in sent_1:
        if w not in stop_words:
            vec_1[total.index(w)]+=1

    for w in sent_2:
        if w not in stop_words:
            vec_2[total.index(w)]+=1


    return 1-cosine_distance(vec_1,vec_2)

def build_matrix(sentences):
    stop_words = stopwords.words('english')

    sim_matrix=np.zeros((len(sentences),len(sentences)))
    ## Adjacency matrix

    for id1 in range(len(sentences)):
        for id2 in range(len(sentences)):
            if id1==id2:  #escaping diagonal elements
                continue
            else:
                sim_matrix[id1][id2]=get_similarity(sentences[id1],sentences[id2],stop_words)

    return sim_matrix

def pagerank(text, eps=0.000001, d=0.85):
    score_mat = np.ones(len(text)) / len(text)
    delta=1
    ### iterative approach
    while delta>eps:
        score_mat_new = np.ones(len(text)) * (1 - d) / len(text) + d * text.T.dot(score_mat)
        delta = abs(score_mat_new - score_mat).sum()
        score_mat = score_mat_new

    return score_mat_new

def summarizer(article,req=3):
    summarized=[]
    nltk.download('punkt')
    nltk.download('stopwords')
    sentence=get_sentences(article)

    sim_matrix=build_matrix(sentence)

    score=pagerank(sim_matrix)

    ranked_sentence = sorted(((score[i],s) for i,s in enumerate(sentence)), reverse=True)

    for i in range(req):
        summarized.append(" ".join(ranked_sentence[i][1]))

    return ' '.join(summarized)

def main():
    with open(sys.argv[1], 'r') as my_file:
        print(summarizer(my_file.read()))

if __name__=="__main__":
    main()

