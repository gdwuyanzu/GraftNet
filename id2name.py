import json
from tqdm import tqdm

entities = 'datasets/webqsp/full/entities.txt'
mid2name = '../mid2name.txt'
entities_name = 'datasets/webqsp/full/entities_name.txt'
entities_name_not_found = 'datasets/webqsp/full/entities_name_not_found.txt'

def _convert_freebase_id(x):
    return "<fb:" + x[1:].replace("/", ".") + ">"

def convert_id_to_name():
    with open(entities) as entities_f, open(mid2name) as mid2name_f, \
        open(entities_name, 'w') as entities_name_f, open(entities_name_not_found, 'w') as entities_name_not_found_f:
        mid2name_map = {}
        for line in tqdm(mid2name_f):
            mid, name = line.strip().split('\t')
            mid = _convert_freebase_id(mid)
            mid2name_map[mid] = name
        total = 0.0
        find = 0.0
        for line in tqdm(entities_f):
            total += 1
            mid = line.strip()
            if mid in mid2name_map:
                find += 1
                entities_name_f.write(mid + ' ' + mid2name_map[mid] + '\n')
            else:
                entities_name_not_found_f.write(mid + '\n')
        print('total entities: ' + total)
        print('find_name_entities: ' + find)
        print('not_find_name_entities: ' + (total - find))

if __name__ == "__main__":
    convert_id_to_name() 