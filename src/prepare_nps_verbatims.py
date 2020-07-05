import json
import gzip
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import npsverbatim_to_aspectsentiment_hr

domains = ["Prism", "telco", "insurance", "retail"]

# path to files
fp = '../data/raw/bain.nosync/'
fn = '_annotation_data_filtered.jsonl'

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def sentence_segment_filter_docs(doc_array):
    sentences = []

    for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_threads=8):
        sentences.append([sent.text.strip() for sent in doc.sents])

    return sentences

# get verbatim documents
verbatims = []
domain_verbatims = {}

print('Loading and Filtering NPS verbatims')
for domain in domains:
    path = fp + domain + fn
    with open(path, "r") as file:
        limit = 1000000
        counter = 0
        for line in file:
            verbatim = json.loads(line)
            if verbatim['text'] != None and len(verbatim['text']) > 0:
                verbatims.append(verbatim['text'])
                # print(review)
                counter += 1
            if counter % 1000 == 0 and counter >= 1000:
                pass #print(counter, end=' ')
            if counter == limit:
                break
            
    print(f'Found {len(verbatims)} NPS verbatims')

    print(f'Tokenizing {domain} NPS verbatims...')

    sentences = sentence_segment_filter_docs(verbatims)
    nr_sents = sum([len(s) for s in sentences])
    print(f'Segmented {nr_sents} verbatim sentences')

    # Save to file
    max_sentences = int(25e6)
    fn_out = f'../data/transformed/{domain}_corpus_{nr_sents}.txt'

    # filter sentences by appearance in the semeval dataset

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

        plt.hist(sentence_counts,density=True, alpha=0.5, histtype='stepfilled', edgecolor='none')
        # counts, bins = np.histogram(list(sentence_counts.keys()), list(sentence_counts.values()))
        # plt.hist(bins[:-1], bins, weights=counts, label=domain, density=True)
        # #plt.hist(sentence_counts.keys(), sentence_counts.values(), density=True, label=domain)
        plt.title(f'Frequency of sentence counts per verbatim for {domain}')
        plt.xlabel('Amount of sentences per verbatim')
        plt.savefig(f'{domain}_sentence_distribution.png')


    print(f'Done writing to {fn_out}')
