from gensim import corpora, models, similarities
import jieba

with open('../data/transformed/Prism_corpus_346226.txt', 'r') as file:
    prism_data = file.read().replace('\n', '')

with open('../data/transformed/retail_corpus_1117942.txt', 'r') as file:
    retail_data = file.read().replace('\n', '')

with open('../data/transformed/telco_corpus_627866.txt', 'r') as file:
    telco_data = file.read().replace('\n', '')

with open('../data/transformed/insurance_corpus_856076.txt', 'r') as file:
    insurance_data = file.read().replace('\n', '')

# texts = [prism_data, retail_data, telco_data, insurance_data]
# keyword = prism_data
# texts = [jieba.lcut(text) for text in texts]
# dictionary = corpora.Dictionary(texts)
# feature_cnt = len(dictionary.token2id)
# corpus = [dictionary.doc2bow(text) for text in texts]
# tfidf = models.TfidfModel(corpus) 
# kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
# index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
# sim = index[tfidf[kw_vector]]
# for i in range(len(sim)):
#     print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))

import spacy

nlp = spacy.load('en_core_web_lg')

def remove_pronoun(text):
    doc = nlp(text.lower())
    result = [token for token in doc if token.lemma_ != '-PRON-']
    return " ".join(result)

def remove_stopwords_fast(text):
    doc = nlp(text.lower())
    result = [token.text for token in doc if token.text not in nlp.Defaults.stop_words]
    return " ".join(result)

def process_text(text):
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    return " ".join(result)


def calculate_similarity(text1, text2):
    base = nlp(process_text(text1))
    compare = nlp(process_text(text2))
    return base.similarity(compare)

doc1 = nlp(insurance_data[:100000])
doc2 = nlp(prism_data[:100000])
# doc3 = nlp(telco_data[:100000])
# doc4 = nlp(retail_data[:100000])

print(calculate_similarity(retail_data[:1000000], prism_data[:1000000]))