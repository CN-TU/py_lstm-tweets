
# MLCS 2019 – Workshop on Machine Learning for CyberSecurity
# Competition on Multi-Task Learning in Natural Language Processing for Cybersecurity Threat Awareness
#
#  "descriptive.py" extracts the entity file with entities and related statistics
#
# TU Wien, Inst. of Telec., CN Group
# FIV, Aug 2019 

from text_processing import tokenize_sentences, extract_words
import numpy as np
import pandas as pd
import fileinput
import sys

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ******* CONFIGURATION OPTIONS *******
HEADER = 1
ENT_FILE = 'obj/entities.csv'

SHOW_VALUES = 0
ENT = 'id'
    
# ******* FUNCTIONS *******
def remove_all(substr, strr):
    index = 0
    length = len(substr)
    while strr.find(substr) != -1:
        index = strr.find(substr)
        strr = strr[0:index] + strr[index+length:]
    return strr

# ******* BEGIN *******

if __name__ == "__main__":
    bow={}

    expendable_words = stopwords.words('english')

    print("Creating entity hash table...")
    for line in fileinput.input(sys.argv[1]):
        if (not fileinput.isfirstline()) or (HEADER==0):
            line = line.rstrip()
            t,r,e,A,B,C = line.split(",")
            r,A,B,C = int(r),int(A),int(B),int(C)
            if r==1:
                e = remove_all("B-",e)
                e = remove_all("I-",e)
                words = extract_words(t)
                entities = extract_words(e)
                for word in words:
                    i = words.index(word)
                    aux = entities[i]
                    if (aux != 'o' and word not in expendable_words):
                        if word not in bow:
                            bow[word]=[0,0,0,0,0,r,0,A,B,C]
                        else:
                            bow[word][5] = bow[word][5]+1
                            bow[word][7] = bow[word][7]+A
                            bow[word][8] = bow[word][8]+B
                            bow[word][9] = bow[word][9]+C
                        if aux=="org":
                            bow[word][0]=bow[word][0]+1
                        elif aux=="pro":
                            bow[word][1]=bow[word][1]+1
                        elif aux=="ver":
                            bow[word][2]=bow[word][2]+1
                        elif aux=="vul":
                            bow[word][3]=bow[word][3]+1
                        elif aux=="id":
                            bow[word][4]=bow[word][4]+1

    for line in fileinput.input(sys.argv[1]):
        if (not fileinput.isfirstline()) or (HEADER==0):
            line = line.rstrip()
            t,r,e,A,B,C = line.split(",")
            r,A,B,C = int(r),int(A),int(B),int(C)
            if r==0:
                words = extract_words(t)
                for word in words:
                    if word in bow:
                        bow[word][6] = bow[word][6]+1
                        bow[word][7] = bow[word][7]+A
                        bow[word][8] = bow[word][8]+B
                        bow[word][9] = bow[word][9]+C

    df = pd.DataFrame.from_dict(bow,orient='index', columns=['org', 'pro', 'ver', 'vul', 'id', 'r', 'no-r', 'A', 'B', 'C'])
                
    if SHOW_VALUES:
        print(df.loc[df[ENT] > 1])

    print("Saving entity file...")
    df.to_csv(ENT_FILE, sep=',')


