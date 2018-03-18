from __future__ import absolute_import, division, print_function
#for preventing version problems between python3 and python2
import codecs
#for reading dataset in utf-8 encoding
import glob
import re
#for use of regular expressions
import logging
import multiprocessing
#to reduce load and execution time
import os
import nltk
#for raw data processing
import gensim.models.word2vec as w2v
#for utilising word2vec algorithm
import sklearn.manifold
#for dimensionality reduction from 1000 to 2 dimensions
import numpy as np
#matrix operations
import matplotlib.pyplot as plt
#plotting graphs
import pandas as pd
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
book_filenames = sorted(glob.glob("data/*.txt"))
#for getting books location

corpus_raw = u""
#adding all the processed data to this string
for book_filename in book_filenames:
    with codecs.open(book_filename, 'r', 'utf-8') as book_file:
        #appending utf data to the corpus
        corpus_raw += book_file.read()


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
#tokenize the obtained data to bytes
def sentence_to_wordlist(raw):
    clean = re.sub('[^a-zA-Z]',' ',raw)
    words = clean.split()
    return words
#cleaning data or corpus
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

#counting number of words in the corpus
token_count = sum([len(sentence) for sentence in sentences])


num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()

context_size = 7
downsampling = 1e-3
seed = 1
w2vec = w2v.Word2Vec(sg=1,seed = seed, workers= num_workers, size = num_features, min_count = min_word_count, window = context_size, sample = downsampling)
w2vec.build_vocab(sentences)
#w2vec.train(sentences, total_examples = len(w2vec.wv.vocab), epochs = 10)


if not os.path.exists('trained'):
    os.makedirs('trained')

#w2vec.save(os.path.join('trained', 'w2vec.w2v'))
w2vec = w2v.Word2Vec.load(os.path.join('trained','w2vec.w2v'))

tsne = sklearn.manifold.TSNE(n_components = 2, random_state=0)
all_word_vectors_matrix = w2vec.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


points = pd.DataFrame([(word, coords[0], coords[1]) for word, coords in [(word, all_word_vectors_matrix_2d[w2vec.wv.vocab[word].index]) for word in w2vec.wv.vocab ]], columns = ['word','x','y'])
print(points.head(10))
sns.set_context('poster')

points.plot.scatter('x','y',s=10, figsize = (20,12))
def plot_region(x_bounds, y_bounds):
    slice = points[(x_bounds[0]<=points.x) & (points.x <= x_bounds[1]) & (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]
    ax = slice.plot.scatter('x','y',s = 35, figsize =(10,8))
    for i,point in slice.iterrows():
        ax.text(point.x + 0.005, point.y +0.005, point.word, fontsize=11)
plot_region(x_bounds=(0,5) , y_bounds=(0,5))
w2vec.most_similar("Krishna")
w2vec.most_similar("Pandavas")
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = w2vec.wv.most_similar_cosmul(positive=[end2, start1],negative=[end1])
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

nearest_similarity_cosmul("Pandavas", "Kauravas", "Draupadi")

