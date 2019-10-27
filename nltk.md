## Tokenize
``` python
from nltk import sent_tokenize, word_tokenize

sentences = sent_tokenize(document)
words = word_tokenize(document)
```

## Count and most common
``` python
from nltk.probability import FreqDist
words_counts = FreqDist(words)

words_counts.items() #unsorted
words_counts.most_common(20) #sorted
```

## Tags
``` python
nltk.help.upenn_tagset('NN') # checks documentation
```



## Bag of Words
``` python
from nltk.probability import FreqDist

def get_vocabulary(words, top=500):
    word_counts = FreqDist(words)
    vocabulary = word_counts.most_common(top)
    return vocabulary

words = []
for d in documents:
    _words = [w.lower() for w in word_tokenize(d) if w.isalpha()]
    words += _words

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

words = [w for w in words if w not in stop_words]

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

words = [stemmer.stem(w) for w in words]

get_vocabulary(words)
```

## Vectorize
``` python
import numpy as np

def vectorize_doc(document, vocabulary):
    doc_words = word_tokenize(document.lower())
    doc_words = [stemmer.stem(word) for word in doc_words]
    doc_frequencies = FreqDist(doc_words)
    voc_size = len(vocabulary)
    doc_vector = np.zeros(voc_size)
    for idx in range(0, voc_size):
        token = vocabulary[idx]
        if token in doc_frequencies.keys():
            doc_vector[idx] = doc_frequencies[token]
    return doc_vector

vocabulary = [w for w, _ in get_vocabulary(words)]

idx = 1467
vectorize_doc(documents[idx], vocabulary)
```