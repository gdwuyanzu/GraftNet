import json
from tqdm import tqdm

id_text_file = './id2text.json'
entities_file = '../model/webqsp/full/entities.txt'
webqsp_subgraphs_file = './webqsp_subgraphs.json'

def get_id_text():
    id2text_map = {}
    with open(webqsp_subgraphs_file) as webqsp_subgraphs_file:
        for line in tqdm(webqsp_subgraphs_file):
            question = json.loads(line)
            for entities in question['entities']:
                entities_text = entities['text']
                entities_id = entities['kb_id']
                id2text_map[entities_id] = entities_text
            for answers in question['answers']:
                entities_text = entities['text']
                entities_id = entities['kb_id']
                id2text_map[entities_id] = entities_text
            for entities in question['subgraph']['entities']:
                entities_text = entities['text']
                entities_id = entities['kb_id']
                id2text_map[entities_id] = entities_text