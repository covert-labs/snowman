import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import tldextract

def strip_tld(domain):
    if domain.startswith('.'):
        domain = domain[1:]

    if ',' in domain:
        domain = domain.split(',')[0]

    e = tldextract.extract(domain)
    if e.subdomain:
        return e.subdomain + '.' + e.domain
    else:
        return e.domain

class DataPrep:
    def __init__(self):
        pass
    
    def load_url_file(self, file_path, skip_lines=0):
        with open(file_path) as file:
            lines = file.readlines()
        raw_url_strings = [strip_tld(line) for line in lines[skip_lines:]]
        return raw_url_strings

    def to_one_hot(self, input_str,max_index=256, padding_length=30):
        """Transform single input string into zero-padded one-hot (index) encoding."""
        input_one_hot = one_hot(" ".join(list(input_str)), n = max_index)
        return pad_sequences([input_one_hot], maxlen=padding_length)
        
    def to_one_hot_array(self, string_list, max_index= 256):
        """Transform list of input strings into numpy array of zero-padded one-hot (index) encodings."""
        self.max_index = max_index
        x_one_hot = [one_hot(" ".join(list(sentence)), n = max_index) for sentence in string_list]
        self.max_len = max([len(s) for s in x_one_hot])
        X = np.array(pad_sequences(x_one_hot, maxlen=self.max_len))
        
        self.relevant_indices = np.unique(X)
        
        charset = set(list(" ".join(string_list)))
        self.charset = charset 
        
        encoding = one_hot(" ".join(charset),n=max_index)
        self.charset_map = dict(zip(charset,encoding) )
        self.inv_charset_map = dict(zip(encoding, charset) )
        
        return X
        
    def shuffle(self, X,Y):
        a = list(range(Y.size))
        np.random.shuffle(a)

        X = X[a]
        Y = Y[a]
        
        return(X,Y)
    
    def train_test_split(self, X,Y,proportion):
        (X,Y) = self.shuffle(X,Y)
        max_ind = int(proportion * X.shape[0])
        return(X[:max_ind,:],X[(max_ind+1):,:],Y[:max_ind,], Y[(max_ind+1):, ])