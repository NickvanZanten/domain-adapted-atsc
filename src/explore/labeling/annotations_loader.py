import json
import pandas as pd
# annotations = pd.read_json('prism_aspectterm_annotations.jsonl', lines=True, orient='records')
annotations = pd.read_json('prism_aspectterm_annotations.jsonl', lines=True, orient='records')

processed = []
for idx, annotation in annotations.iterrows():
    if annotation['answer'] == 'accept':
        for term in annotation['spans']:
            aspectTerm = ''
            if term['token_end'] > term['token_start']:
                for i in range(0, term['token_end'] - term['token_start'] + 1): # Walk through array of tokens
                    aspectTerm = aspectTerm + annotation['tokens'][term['token_start'] + i]['text']
                    if i != term['token_end']:
                        aspectTerm = aspectTerm + ' '
            else:
                aspectTerm = annotation['tokens'][term['token_start']]['text']

            processed.append({'Country': annotation['Country'],
                    'NPS': annotation['NPS'], 'text': annotation['text'], 'aspectTerm': aspectTerm})

df = pd.DataFrame(processed)
df.to_json('processed_annotations.jsonl', lines=True, orient='records')
