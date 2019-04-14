import os
from tqdm import tqdm
import json

mid_name_map = {}
neigh_dir_path = '/home/user_data55/zhengnt/GraftNet/preprocessing/freebase_2hops/stagg.neighborhoods/'
data_dir_path = '/home/user_data55/zhengnt/GraftNet/datasets/webqsp/full/'

def _filter_relation(relation):
    if relation == "<fb:type.object.name>":
        return True
    else:
        return False

def _filter_mid(mid):
    if mid[:4] == "<fb:":
        return False
    else:
        return True

def getMid2NameMap():
    files = os.listdir(neigh_dir_path)
    for file in files:
        with open(neigh_dir_path + file) as f:
            for line in tqdm(f):
                try:
                    e1, rel, e2 = line.strip().split(None, 2)
                except ValueError:
                    continue
                if _filter_relation(rel):
                    mid_name_map[e1] = e2

def getString2StringMap():
    train_file, test_file = data_dir_path + 'train.json', data_dir_path + 'test.json'
    with open(train_file) as train_f:
        for line in train_f:
            question = json.loads(line)
            entities = question['subgraph']['entities']
            for entity in entities:
                if _filter_mid(entity['kb_id']):
                    mid_name_map[entity['kb_id']] = entity['kb_id']
    with open(test_file) as test_f:
        for line in test_f:
            question = json.loads(line)
            entities = question['subgraph']['entities']
            for entity in entities:
                if _filter_mid(entity['kb_id']):
                    mid_name_map[entity['kb_id']] = entity['kb_id']

def writeMap():
    mid2name = data_dir_path + 'mid2name'
    with open(mid2name, 'w') as mid_f:
        for k, v in mid_name_map.items():
            mid_f.write(json.dumps({k:v}) + '\n')

def analysis_type():
    train_file, test_file, mid2name = data_dir_path + 'train.json', data_dir_path + 'test.json', data_dir_path + 'mid2name'
    with open(mid2name) as name_f:
        for line in tqdm(name_f):
            line = json.loads(line)
            for k,v in line.items():
                mid_name_map[k] = v
    train_entity_total, train_entity_hit, test_entity_total, test_entity_hit = 0.0, 0.0, 0.0, 0.0
    with open(train_file) as train_f, open(test_file) as test_f:
        for line in tqdm(train_f):
            question = json.loads(line)
            entities = question['subgraph']['entities']
            for entity in entities:
                train_entity_total += 1
                if entity['kb_id'] in mid_name_map:
                    train_entity_hit += 1
        for line in tqdm(test_f):
            question = json.loads(line)
            entities = question['subgraph']['entities']
            for entity in entities:
                test_entity_total += 1
                if entity['kb_id'] in mid_name_map:
                    test_entity_hit += 1
    print("train_entity_total: " + train_entity_total)
    print("train_entity_hit: " + train_entity_hit)
    print(train_entity_hit / train_entity_total)
    print("test_entity_total: " + test_entity_total)
    print("test_entity_hit: " + test_entity_hit)
    print(test_entity_hit / test_entity_total)

if __name__ == "__main__":
    # getMid2NameMap()
    # getString2StringMap()
    # writeMap()
    analysis_type()