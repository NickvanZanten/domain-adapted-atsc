import xml.etree.ElementTree as ET


def semeval2014term_to_aspectsentiment_hr(filename, remove_conflicting=True):
    sentimap = {
        'positive': 'POS',
        'negative': 'NEG',
        'neutral': 'NEU',
        'conflict': 'CONF',
    }

    def transform_aspect_term_name(se):
        return se

    with open(filename) as file:

        sentence_elements = ET.parse(file).getroot().iter('sentence')

        sentences = []
        aspect_term_sentiments = []
        classes = set([])

        for j, s in enumerate(sentence_elements):
            # review_text = ' '.join([el.text for el in review_element.iter('text')])

            sentence_text = s.find('text').text
            aspect_term_sentiment = []
            for o in s.iter('aspectTerm'):
                aspect_term = transform_aspect_term_name(o.get('term'))
                classes.add(aspect_term)
                sentiment = sentimap[o.get('polarity')]
                if sentiment != 'CONF':
                    aspect_term_sentiment.append((aspect_term, sentiment))
                else:
                    if remove_conflicting:
                        pass
                        # print('Conflicting Term found! Removed!')
                    else:
                        aspect_term_sentiment.append((aspect_term, sentiment))

            if len(aspect_term_sentiment) > 0:
                aspect_term_sentiments.append(aspect_term_sentiment)
                sentences.append(sentence_text)

        cats = list(classes)
        cats.sort()

    idx2aspectlabel = {k: v for k, v in enumerate(cats)}
    sentilabel2idx = {"NEG": 1, "NEU": 2, "POS": 3, "CONF": 4}
    idx2sentilabel = {k: v for v, k in sentilabel2idx.items()}

    return sentences, aspect_term_sentiments, (idx2aspectlabel, idx2sentilabel)

def semeval2016category_to_aspectsentiment_hr(filename, remove_conflicting=True):
    sentimap = {
        'positive': 'POS',
        'negative': 'NEG',
        'neutral': 'NEU',
        'conflict': 'CONF',
    }

    def transform_aspect_category_name(se):
        return se

    with open(filename) as file:

        sentence_elements = ET.parse(file).getroot().iter('sentence')

        sentences = []
        aspect_category_sentiments = []
        classes = set([])

        for j, s in enumerate(sentence_elements):
            # review_text = ' '.join([el.text for el in review_element.iter('text')])

            sentence_text = s.find('text').text
            aspect_category_sentiment = []
            for o in s.iter('aspectCategory'):
                aspect_category = transform_aspect_category_name(o.get('category'))
                classes.add(aspect_category)
                sentiment = sentimap[o.get('polarity')]
                if sentiment != 'CONF':
                    aspect_category_sentiment.append((aspect_category, sentiment))
                else:
                    if remove_conflicting:
                        pass
                        # print('Conflicting Term found! Removed!')
                    else:
                        aspect_category_sentiment.append((aspect_category, sentiment))

            if len(aspect_category_sentiment) > 0:
                aspect_category_sentiments.append(aspect_category_sentiment)
                sentences.append(sentence_text)

        cats = list(classes)
        cats.sort()

    idx2aspectlabel = {k: v for k, v in enumerate(cats)}
    sentilabel2idx = {"NEG": 1, "NEU": 2, "POS": 3, "CONF": 4}
    idx2sentilabel = {k: v for v, k in sentilabel2idx.items()}

    return sentences, aspect_category_sentiments, (idx2aspectlabel, idx2sentilabel)