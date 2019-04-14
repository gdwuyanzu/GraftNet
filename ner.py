import os
import nltk
from nltk.tag import StanfordNERTagger
from tqdm import tqdm

# java_path = r'D:\java\jdk1.8\bin\java.exe;F:\stanford-ner-2015-12-09\stanford-ner.jar;F:\stanford-ner-2015-12-09\lib\slf4j-api.jar'
# os.environ['JAVAHOME'] = java_path
# path_to_jar='/user_data/stanford-ner-2018-10-16/stanford-ner.jar'

tagger = StanfordNERTagger(model_filename = r'F:\stanford-ner-2015-12-09\classifiers\english.muc.7class.distsim.crf.ser.gz', \
    path_to_jar = r'F:\stanford-ner-2015-12-09\stanford-ner.jar')
mid2name = r'C:\Users\znt\Desktop\mid2name'
mid2name2type = r'C:\Users\znt\Desktop\mid2name2type'

tag_type_map = {'PERSON': 'who', 'DATE': 'when', 'LOCATION': 'where', 'ORGANIZATION': 'what', 'Time': 'when', 'Percent': 'how'}
def getNameType(name):
    name = name.split()
    tag_name = tagger.tag(name)
    tag = None
    for a, b in tag_name:
        if b != 'O':
            return

def getMidType():
    with open(mid2name) as name_f:
        for line in tqdm(name_f):
            mid, name = line.strip().split(None, 1)

sen = 'Shantou'
print(tagger.tag(sen.split()))
# entities = []
# print(tag_sen)

# for term, tag in tag_sen:
#     temp_entity_name = ''
#     temp_named_entity = None
#     if tag != '0':
#         temp_entity_name = ' '.join([temp_entity_name, term]).strip()
#         temp_named_entity = (temp_entity_name, tag)
#     else:
#         if temp_named_entity:
#                entities.append(temp_named_entity)
#                temp_entity_name = ''
#                temp_named_entity = None

# print(entities)