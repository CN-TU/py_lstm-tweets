
# MLCS 2019 – Workshop on Machine Learning for CyberSecurity
# Competition on Multi-Task Learning in Natural Language Processing for Cybersecurity Threat Awareness

# Solution based on LSTM
# TU Wien, Inst. of Telec., CN Group
# FIV, Aug 2019 

import numpy as np
import pandas as pd
import fileinput
import sys
import re
import csv
import collections
import pickle

from keras.preprocessing import sequence 
from keras.models import Sequential, load_model 
from keras.layers import Dense, Dropout, Embedding, LSTM 

from text_processing import tokenize_sentences, extract_words
from tweet_manager import extract_train_bow, extract_id_keys, init_val, display_tweet_info, display_osint, extract_osint, check_id_keys, update_validation, display_prediction, output_validation


# ******* USE HELP *******

TRAINING, CONFIG_FILE, REQ = 0, 0, 0
for arg in sys.argv:
    if arg == "-t": 
        i = sys.argv.index(arg)
        training_file = sys.argv[i+1]
        TRAINING, REQ = 1, 1
    elif arg == "-e": 
        i = sys.argv.index(arg)
        test_file = sys.argv[i+1]
        REQ = 1
    elif arg == "-c": 
        i = sys.argv.index(arg)
        config_file = sys.argv[i+1]
        CONFIG_FILE = 1

if REQ == 0:
    print("\n use: python3 lstm_based_classif.py -t [t_file] -e [e_file] -c [c_file] \n")
    print("\t t_file: CSV file for model training. \t \t Format: <tweet>,<relevance>,<entities>,<A>,<B>,<C>")
    print("\t e_file: CSV file for model test/eval. \t Format: <tweet>,<relevance>,<entities>,<A>,<B>,<C> or simply: <tweet> ")
    print("\t c_file:  TXT file with configuration values. \n")
    print("\t A t_file or a e_file is required. \n")    
    quit()
            
# ******* CONFIGURATION OPTIONS *******
    
config_options_flags={
    'HEADER_TRAIN':1,       # 1 if the training_file has header  
    'HEADER_TEST':0,        # 1 if the test_file has header
    'OUT2FILE':1,           # 1 if the results/outputs are to be saved in a CSV file
    'EVAL':0,               # 1 if the test_file must be evaluated (data for validation is required) 
    'VERBOSE':0,            # 0: only predictions, 1: predictions and real (if EVAL), 2: display complete info (if EVAL), 3: display only evaluation results (if EVAL)
    'SAVE_MODEL':0}         # 1 saves trained models

config_options_strings={
    'ENT_FILE':"obj/entities.csv",      # file with entities tables for entity prediction
    'OUTPUT_FILE':"results.csv",        # file to save outputs/results (if OUT2FILE)
    'R_MOD_FILE':"obj/model_r.h5",      # file with the LSTM model for "relevance" prediction 
    'A_MOD_FILE':"obj/model_A.h5",      # file with the LSTM model for "A" prediction
    'B_MOD_FILE':"obj/model_B.h5",      # file with the LSTM model for "B" prediction
    'C_MOD_FILE':"obj/model_C.h5",      # file with the LSTM model for "C" prediction
    'DICT_FILE':"obj/lstm_dict.pkl"}    # file with the LSTM-dictionary

# Read configuration
if CONFIG_FILE:
    for line in fileinput.input(config_file):
        name,val=line.split(":")
        if (name in config_options_flags):
            config_options_flags[name]=int(val.rstrip())
        if (name in config_options_strings):
            config_options_strings[name]=val.rstrip()
           
if config_options_flags['EVAL'] == 0:
    config_options_flags['VERBOSE'] = 0
    
HEADER_TRAIN = config_options_flags['HEADER_TRAIN']
HEADER_TEST  = config_options_flags['HEADER_TEST']
OUT2FILE     = config_options_flags['OUT2FILE']
EVAL         = config_options_flags['EVAL']
VERBOSE      = config_options_flags['VERBOSE']
SAVE_MODEL   = config_options_flags['SAVE_MODEL']

ENT_FILE    = config_options_strings['ENT_FILE']
OUTPUT_FILE = config_options_strings['OUTPUT_FILE']
R_MOD_FILE  = config_options_strings['R_MOD_FILE']
A_MOD_FILE  = config_options_strings['A_MOD_FILE']
B_MOD_FILE  = config_options_strings['B_MOD_FILE']
C_MOD_FILE  = config_options_strings['C_MOD_FILE']
DICT_FILE   = config_options_strings['DICT_FILE']

#print("\nConfiguration options:")
#print(config_options_flags)
#print(config_options_strings)
#print("\n")


# ******* FUNCTIONS *******   

# Convert text
def convert_text(text):
    text_list = text.split(' ')
    return [vocab_text[t]+1 for t in text_list]

def extract_counts(words):
    vec=[]
    for word in words:
        if word in dictionary:
            vec.append(dictionary[word])
        else:
            vec.append(0)            
    return vec

def init_model(model):
    model.add(Embedding(num_words, 50, input_length=60)) 
    model.add(Dropout(0.2)) 
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
    model.add(Dense(250, activation='relu')) 
    model.add(Dropout(0.2)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

# ******* BEGIN *******

# We load pre-analyzed words obtained with "descriptive.py"
train_bow = extract_train_bow(ENT_FILE)
id_keys = extract_id_keys(train_bow)

# Variable/model initialization
if EVAL: valDL = init_val()    
model_r,model_A,model_B,model_C = Sequential(),Sequential(),Sequential(),Sequential() 

# Training 
if TRAINING:
    all_words, sentences, y_r, y_A, y_B, y_C = [], [], [], [], [], []

    if VERBOSE==2: print("Reading training tweets...")
    for line in fileinput.input(training_file):
        if (not fileinput.isfirstline()) or (HEADER_TRAIN==0):
            line = line.rstrip()
            t,r,e,A,B,C = line.split(",")
            r,A,B,C = int(r),int(A),int(B),int(C)
            all_words = all_words + extract_words(t)
            sentences.append(t)
            y_r.append(r)
            y_A.append(A)
            y_B.append(B)
            y_C.append(C)
            
    vectorizer = collections.Counter(all_words).most_common()
    dictionary = dict()
    for word, _ in vectorizer:
        dictionary[word] = len(dictionary)

    num_words = len(dictionary)
    if SAVE_MODEL:
        save_obj(dictionary,DICT_FILE)

    df = pd.DataFrame(sentences, columns=['tweet'])
    df['tweet'] = df['tweet'].apply(extract_words)
    df['tweet'] = df['tweet'].apply(extract_counts)

    y_r,y_A,y_B,y_C = np.array(y_r),np.array(y_A),np.array(y_B),np.array(y_C)
    X = np.array(df['tweet'])

    # Define network architecture and compile 
    print("Creating LSTM models...")
    model_r,model_A,model_B,model_C = init_model(model_r), init_model(model_A), init_model(model_B), init_model(model_C) 

    print("Training LSTM models...")
    X = sequence.pad_sequences(X, maxlen=60)
    model_r.fit(X, y_r, batch_size=64, epochs=5) 
    model_A.fit(X, y_A, batch_size=64, epochs=5) 
    model_B.fit(X, y_B, batch_size=64, epochs=5) 
    model_C.fit(X, y_C, batch_size=64, epochs=5) 
    if SAVE_MODEL:
        model_r.save(R_MOD_FILE)
        model_A.save(A_MOD_FILE)
        model_B.save(B_MOD_FILE)
        model_C.save(C_MOD_FILE)
else:
        print("Loading LSTM models...")
        model_r = load_model(R_MOD_FILE)
        model_A = load_model(A_MOD_FILE)
        model_B = load_model(B_MOD_FILE)
        model_C = load_model(C_MOD_FILE)
        dictionary = load_obj(DICT_FILE)
        num_words = len(dictionary)
    

if OUT2FILE: sys.stdout = open(OUTPUT_FILE, 'w', encoding='utf-8')
if VERBOSE<2:
    print("id, ", end='', flush=True)
    if VERBOSE:  print("clean_tweet, relevance (real), A (real), B (real), C (real), ", end='', flush=True)
    print("relevance (pred), A (pred), B (pred),  C (pred), ", end='', flush=True)
    if VERBOSE:  print("entitites (real), ", end='', flush=True)
    print("entitites (pred)", end='', flush=True)

if VERBOSE==2: print("Reading test tweets...")
test_tweets = 0
for line in fileinput.input(test_file):
    if (not fileinput.isfirstline()) or (HEADER_TRAIN==0):
        test_tweets = test_tweets +1
        line = line.rstrip()       
        if EVAL:
            t,r,e,A,B,C=line.split(",")
            r,A,B,C = int(r),int(A),int(B),int(C)
        else:
            t = line
            r,e,A,B,C=0,0,0,0,0
        sentence = extract_words(t)
       
        display_tweet_info(test_tweets,t,r,A,B,C,VERBOSE)
                    
        xt = [extract_counts(extract_words(t))]
        xt = sequence.pad_sequences(xt, maxlen=60)
        y_r = model_r.predict_classes(xt)
        y_A = model_A.predict_classes(xt)
        y_B = model_B.predict_classes(xt)
        y_C = model_C.predict_classes(xt)
        y_r=y_r[0][0]
        y_A=y_A[0][0]
        y_B=y_B[0][0]
        y_C=y_C[0][0]
        
        if EVAL: valDL = update_validation(valDL,r,A,B,C,y_r,y_A,y_B,y_C)
        display_prediction("Machine learning (feat. vec.)",y_r,y_A,y_B,y_C, VERBOSE)

        pe = extract_osint(t,train_bow,id_keys) if y_r else []
                
        if VERBOSE:     display_osint(e, "(REAL)", VERBOSE, 0)
        display_osint(pe,"(Pred)", VERBOSE, 1)
                                        
        if VERBOSE>1: output_validation(valDL, "LSTM")
        if VERBOSE==2: print("\n")

