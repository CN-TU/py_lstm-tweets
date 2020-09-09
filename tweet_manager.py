
# MLCS 2019 – Workshop on Machine Learning for CyberSecurity
# Competition on Multi-Task Learning in Natural Language Processing for Cybersecurity Threat Awareness

# Functions to extract and display tweet and tweet analysis information 
# TU Wien, Inst. of Telec., CN Group
# FIV, Aug 2019 

import numpy as np
import pandas as pd
import fileinput
import sys
import re
import csv
from datasketch import MinHash

from text_processing import tokenize_sentences, extract_words
            
# ******* FUNCTIONS *******   

def extract_id_keys(train_bow):
    regex = re.compile('[^a-zA-Z]')
    id_keys={}
    for word in train_bow:
        if int(train_bow[word][4])>1:
            #if row['r']>row['no-r']:
            id_keys[regex.sub('', word)]=1
    id_keys = list(id_keys.keys())
    id_keys = list(filter(None, id_keys))
    return id_keys

def check_id_keys(word, id_keys):
    p=0
    t="O"
    a=word.count("-")
    b=word.count(":")
    b2=word.count(".")
    c=any(char.isdigit() for char in word)
    regex = re.compile('[^a-zA-Z]')
    d=regex.sub('', word) in id_keys
    if ((a or b) and c):
        t="ID"
        p=0.6
        if d:
            p=0.8
    elif (b2 and c):
        t="VER"
    return p,t

def extract_osint(sentence,train_bow,id_keys):
    words = extract_words(sentence)
    keys=['ORG','PRO','VER','VUL','ID']
    code = []
    for word in words:
        if word in train_bow:
            aux = [int(train_bow[word][0]),int(train_bow[word][1]),int(train_bow[word][2]),int(train_bow[word][3]),int(train_bow[word][4])]
            if (sum(aux)) > 0:
                pos = aux.index(max(aux))
                code.append(keys[pos])
            else:
                code.append("O")
        else:
            p,t = check_id_keys(word, id_keys)
            code.append(t)

    clone = code.copy();
    
    if code[0]!="O":
        code[0] = "B-" + code[0]
    for i in range(1, len(code)):
        if code[i]!="O":
            if clone[i-1]==code[i]:
                code[i]="I-"+code[i]
            else:
                code[i]="B-"+code[i]
    return " ".join(code) 

def predict_classes_MH(df_small):
    aux = sum(df_small['sim'].values)
    mr = int(sum(df_small['relevance'].values*df_small['sim'].values)/aux>0.5) if max(df_small['sim'].values)>REL_POS_SIM_TH else 0
    mA = int(sum(df_small['A'].values*df_small['sim'].values)/aux>0.5)
    mB = int(sum(df_small['B'].values*df_small['sim'].values)/aux>0.5)
    mC = int(sum(df_small['C'].values*df_small['sim'].values)/aux>0.5)
    return mr,mA,mB,mC
        
def output_validation(val,method):
    #-- Meaning of... val['...']=[TN,FN,FP,TP]
    tpr_rel = val['rel'][3]/(val['rel'][3]+val['rel'][1])  if (val['rel'][3]+val['rel'][1]) != 0 else 1
    tnr_rel = val['rel'][0]/(val['rel'][0]+val['rel'][2])  if (val['rel'][0]+val['rel'][2]) != 0 else 1
    tpr_A = val['A'][3]/(val['A'][3]+val['A'][1]) if (val['A'][3]+val['A'][1]) != 0 else 1
    tnr_A = val['A'][0]/(val['A'][0]+val['A'][2]) if (val['A'][0]+val['A'][2]) != 0 else 1
    tpr_B = val['B'][3]/(val['B'][3]+val['B'][1]) if (val['B'][3]+val['B'][1]) != 0 else 1
    tnr_B = val['B'][0]/(val['B'][0]+val['B'][2]) if (val['B'][0]+val['B'][2]) != 0 else 1
    tpr_C = val['C'][3]/(val['C'][3]+val['C'][1]) if (val['C'][3]+val['C'][1]) != 0 else 1
    tnr_C = val['C'][0]/(val['C'][0]+val['C'][2]) if (val['C'][0]+val['C'][2]) != 0 else 1
    print("\t [TP,TN,FP,FN], Rel[%d, %d, %d, %d],  A[%d, %d, %d, %d], B[%d, %d, %d, %d], C[%d, %d, %d, %d];\t F1(Rel,A,B,C): %.2f, %.2f, %.2f, %.2f; \t %s " 
        % (val['rel'][3], val['rel'][0], val['rel'][2], val['rel'][1],  
           val['A'][3], val['A'][0], val['A'][2], val['A'][1],  
           val['B'][3], val['B'][0], val['B'][2], val['B'][1],  
           val['C'][3], val['C'][0], val['C'][2], val['C'][1], 
           2*tpr_rel*tnr_rel/(tpr_rel+tnr_rel), 2*tpr_A*tnr_A/(tpr_A+tnr_A), 2*tpr_B*tnr_B/(tpr_B+tnr_B), 2*tpr_C*tnr_C/(tpr_C+tnr_C), method), end='', flush=True)

def update_validation(val,r,A,B,C,mr,mA,mB,mC):
    #-- Meaning of... val['...']=[TN,FN,FP,TP]
    pos = r + 2*mr
    val['rel'][pos] = val['rel'][pos]+1
    pos = A + 2*mA
    val['A'][pos] = val['A'][pos]+1 
    pos = B + 2*mB
    val['B'][pos] = val['B'][pos]+1 
    pos = C + 2*mC
    val['C'][pos] = val['C'][pos]+1 
    return val
               
def display_tweet_info(n,t,r,A,B,C,f):

    print("\n%d, " % (n), end='', flush=True)
    if f==1:
        print("%s, %d, %d, %d, %d, " % (t,r,A,B,C), end='', flush=True)
    elif f==2:
        print("NEW TWEET: ", t)
        print("\t (REAL) Relevant:", r,", A:", A,", B:", B,", C:", C)
    
def display_prediction(method,r,A,B,C,f):
    if f<2:
        print("%d, %d, %d, %d, " % (r,A,B,C), end='', flush=True)
    elif f==2:
        print("\t (Pred) Relevant:", r,", A:", A,", B:", B,", C:", C, "\t", method)

def display_osint(e, label, f, last):
    if f<2:
        if last:    print("%s" % (e), end='', flush=True)
        else:       print("%s, " % (e), end='', flush=True)
    elif f==2:
        print("\t",label,"OSINT:", e)
    
def update_stats(s,r,A,B,C):
    s['tweets'] = s['tweets'] + 1
    s['rel'] = s['rel'] + r
    s['A'] = s['A'] + A 
    s['B'] = s['B'] + B 
    s['C'] = s['C'] + C
    s['AB'] = s['AB'] + A * B
    s['AC'] = s['AC'] + A * C 
    s['BC'] = s['BC'] + B * C 
    s['ABC'] = s['ABC'] + A * B * C
    return s

def init_val():
    #-- Validation
    #-- Meaning of... v['...']=[TN,FN,FP,TP]
    v = {}
    v['rel']=[0,0,0,0]
    v['A']=[0,0,0,0]
    v['B']=[0,0,0,0]
    v['C']=[0,0,0,0]
    #-- TPR = TP/(TP+FN); TNR = TN/(TN+FP) ; F1 = 2 * TNR * TPR / (TNR+TPR) 
    return v

def init_stats():
    #-- Counters
    s = {}
    s['tweets'], s['rel'], s['A'], s['B'], s['C'] = 0, 0, 0, 0, 0
    s['ABC'], s['AB'], s['AC'], s['BC'] = 0, 0, 0, 0
    return s

def extract_train_bow(inputfile):
    with open(inputfile, 'r') as f:
        reader = csv.reader(f)
        t = list(reader)
    train_bow = {x[0]: x[1::] for x in t}
    del t,train_bow[""]
    return train_bow

def train_classifiers(vecs, c, n, t):
    df = pd.DataFrame.from_records(vecs, columns=c)
    y_r, y_A, y_B, y_C = df['rel'].values, df['A'].values, df['B'].values, df['C'].values
    X = df.iloc[:,0:n].values
    if t=="RF":    
        from sklearn.ensemble import RandomForestClassifier
        clf_r = RandomForestClassifier(n_estimators=100, random_state=0)
        clf_A = RandomForestClassifier(n_estimators=100, random_state=0)
        clf_B = RandomForestClassifier(n_estimators=100, random_state=0)
        clf_C = RandomForestClassifier(n_estimators=100, random_state=0)
    elif t=="NB":
        from sklearn.naive_bayes import GaussianNB
        clf_r = GaussianNB()
        clf_A = GaussianNB()
        clf_B = GaussianNB()
        clf_C = GaussianNB()
    clf_r.fit(X, y_r)
    clf_A.fit(X, y_A)
    clf_B.fit(X, y_B)
    clf_C.fit(X, y_C)
    return clf_r,clf_A,clf_B,clf_C

