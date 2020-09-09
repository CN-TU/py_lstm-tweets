
#import re
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')

# ******* FUNCTIONS *******
def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)        
    words = sorted(list(set(words)))
    return words

def extract_words(sentence):
    #ignore_words = stopwords.words('english') if include_stopwords else []
    ignore_words = []
    #words = re.sub("[^\w]", " ",  sentence).split() 
    words = sentence.split()
    words_cleaned = [w.lower() for w in words if w not in ignore_words]    
    return words_cleaned    
    
