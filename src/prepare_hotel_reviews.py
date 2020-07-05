import json
import gzip
import spacy
from tqdm import tqdm
from utils import semeval2014term_to_aspectsentiment_hr
import pandas as pd
import json
import gzip
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import npsverbatim_to_aspectsentiment_hr
from langdetect import detect

# path to files
fp = '../data/raw/'
fn = 'hotel_review.txt'
# fn = 'Datafiniti_Hotel_Reviews.csv'
# fn2 = 'Datafiniti_Hotel_Reviews_Jun19.csv'

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def sentence_segment_filter_docs(doc_array):
    sentences = []

    for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_threads=8):
        sentences.append([sent.text.strip() for sent in doc.sents])

    return sentences

# get review documents
reviews = []

print('Loading and Filtering Hotel reviews')
df = pd.read_json(fp + fn, lines=True)
limit = 1000000
counter = 0
for index, row in tqdm(df.iterrows()):
    if row['text'] != None and len(row['text']) > 0:
        try:
            lang = detect(row['text'])
        except:
            continue
        if lang == 'en':
            reviews.append(str(row['text']))
            counter += 1
            
        if counter % 1000 == 0 and counter >= 1000:
            pass #print(counter, end=' ')
        if counter == limit:
            break
        
print(f'Found {len(reviews)} reviews')

print(f'Example of reviews: \n {reviews[:5]}')

print(f'Tokenizing Hotel reviews...')

sentences = sentence_segment_filter_docs(reviews)
nr_sents = sum([len(s) for s in sentences])
print(f'Segmented {nr_sents} verbatim sentences')

# Save to file
max_sentences = int(25e6)
fn_out = f'../data/transformed/hotel_corpus_{nr_sents}.txt'

# filter sentences by appearance in the hotel dataset

removed_reviews_count = 0
with open(fn_out, "w") as f:
    sent_count = 0
    sentence_counts = []

    for sents in tqdm(sentences):
        real_sents = []
        for s in sents:
            x = s.replace(' ', '').replace('\n', '')
            if x != '':
                s_sanitized = s.replace('\n', '')
                real_sents.append(s_sanitized)

        sentence_counts.append(len(real_sents))

        if len(real_sents) >= 2:
            sent_count += len(real_sents)
            str_to_write = "\n" + "\n".join(real_sents) + "\n"
            f.write(str_to_write)

        if sent_count >= max_sentences:
            break

print(f'Done writing to {fn_out}')