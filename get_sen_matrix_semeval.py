import re
import pickle as pk
import csv
from tqdm import tqdm
from pathlib import Path

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def read_data(filename, outfile, sen_len=400):
    reader = open(filename)
    with open('w2v/model/nce_embeddings.pkl','rb') as f:
      emb = pk.load(f)
    with open('w2v/model/nce_dict.pkl','rb') as f:
      w_dict = pk.load(f)
    vars = open(outfile,'wb')
    for elem in tqdm(reader):
      datos = elem.split('\t')
      text = datos[5]
      label = datos[4]
      sen_matrix = []
      for word in preprocess(text):
        if word.startswith('@'):
          word = '<USER/>'
        elif word.startswith('http'):
          word = '<URL/>'
        elif word.startswith('#'):
          word = '<HASHTAG/>'
        else:
          word = word.lower()
        if word in w_dict:
          sen_matrix += [ emb[ w_dict[ word ] ] ]
        else:
          sen_matrix += [ emb[ w_dict[ 'UNK' ] ] ]
      missing = sen_len - len(sen_matrix)
      for x in range(missing):
        sen_matrix += [ emb[ w_dict[ 'UNK' ] ] ] ## embeddings of len
      pk.dump([[sen_matrix],[label.strip()]], vars)



read_data(filename='data/downloaded-avail-a.ds', outfile='data/temp_semeval_mat.pkl')
