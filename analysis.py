import json
from tqdm import tqdm

train_file = "./datasets/webqsp/full/train.json"
dev_file = "./datasets/webqsp/full/dev.json"
test_file = "./datasets/webqsp/full/test.json"

pred_hybrid_file = "./model/webqsp/pred_hybrid"
pred_kb_file = "./model/webqsp/pred_kb"
pred_doc_file = "./model/webqsp/pred_doc"

hybrid_res = "./model/webqsp/ana_res/pred_hybrid_ana"
kb_res = "./model/webqsp/ana_res/pred_kb_ana"
doc_res = "./model/webqsp/ana_res/pred_doc_ana"

hybrid_type_res = "./model/webqsp/ana_res/hybrid_type_res"
kb_type_res = "./model/webqsp/ana_res/kb_type_res"
doc_type_res = "./model/webqsp/ana_res/doc_type_res"

# hybrid_bad = "./model/webqsp/ana_res/hybrid_bad"
hybrid_bad = '/user_data/hybrid_bad'
hybrid_bad_only = '/user_data/hybrid_bad_only'

hybrid_bad_what = '/user_data/hybrid_bad_what'
hybrid_bad_when = '/user_data/hybrid_bad_when'
hybrid_bad_which = '/user_data/hybrid_bad_which'
hybrid_bad_how = '/user_data/hybrid_bad_how'
hybrid_bad_who = '/user_data/hybrid_bad_who'
hybrid_bad_where = '/user_data/hybrid_bad_where'

def get_question_type(file):
    question_type = {"where":0,"what":0,"which":0,"how":0,"when":0,"who":0}
    total = 0
    with open(file) as f:
        for line in tqdm(f):
            total += 1
            question = json.loads(line)
            question_text = question["question"]
            question_word = question_text.split()[0]
            if question_word in question_type:
                question_type[question_word] += 1
            else:
                question_word = question_text.split()[1]
                if question_word in question_type:
                    question_type[question_word] += 1
                else: 
                    question_type[question_text] = 1

    for question_word,cnt in question_type.items():
        print(question_word,cnt,float(cnt) / total)

def get_res_dis():
    question_type_map = {"where": [[],[],[],[]], "what": [[],[],[],[]], "which": [[],[],[],[]], "how":[[],[],[],[]], \
        "when": [[],[],[],[]], "who": [[],[],[],[]]}
    with open(hybrid_res) as f, open(hybrid_type_res, 'w') as res_f: 
        for line in tqdm(f):
            res = json.loads(line)
            question_text = res["question"]
            question_word = question_text.split()[0] if question_text.split()[0] in question_type_map else question_text.split()[1]
            question_type_map[question_word][0].append(res['p'])
            question_type_map[question_word][1].append(res['r'])
            question_type_map[question_word][2].append(res['f1'])
            question_type_map[question_word][3].append(res['hits'])
        for k,v in question_type_map.items():
            for i in range(4):
                question_type_map[k][i] = sum(question_type_map[k][i]) / len(question_type_map[k][i])
        res_f.write(json.dumps(question_type_map) + '\n')

def get_bad_question(f1_bound):
    with open(hybrid_res) as res_f, open(hybrid_bad, 'w') as bad_f:
        for line in tqdm(res_f):
            res = json.loads(line)
            f1 = res['f1']
            if f1 <= f1_bound:
                bad_f.write(json.dumps({'id': res['id'], 'question': res['question'], 'answers': res['answers'], 'f1': res['f1'], 'p': res['p'], \
                    'r': res['r'], 'dist': res['dist']}) + '\n')

def get_bad_only():
    with open(hybrid_bad) as bad_f, open(hybrid_bad_only, 'w') as bad_only_f:
        for line in tqdm(bad_f):
            res = json.loads(line)
            bad_only_f.write(json.dumps({'id': res['id'], 'question': res['question'], 'answers': res['answers'], 'f1': res['f1'], 'p': res['p'], \
                    'r': res['r']}) + '\n')

def get_bad_type():
    with open(hybrid_bad) as bad_f, open(hybrid_bad_what, 'w') as what_f, open(hybrid_bad_how, 'w') as how_f, \
        open(hybrid_bad_when, 'w') as when_f, open(hybrid_bad_which, 'w') as which_f, open(hybrid_bad_who, 'w') as who_f, \
            open(hybrid_bad_where, 'w') as where_f:
        question_type_map = {"where": where_f, "what": what_f, "which": which_f, "how": how_f, "when": when_f, "who": who_f}
        for line in tqdm(bad_f):
            res = json.loads(line)
            question_text = res["question"]
            question_word = question_text.split()[0] if question_text.split()[0] in question_type_map else question_text.split()[1]
            question_type_map[question_word].write(json.dumps(res) + '\n')    

def comRecall(answer_id,entity_id):
    total = len(answer_id)
    hit = 0.0
    for answer in answer_id:
        if answer in entity_id:
            hit += 1
    if total == 0:
        return 0
    else:
        return hit / total

def anaRecall():
    question_type = {"where":[], "what": [], "which": [], "how": [], "when": [], "who": []}
    with open(train_file) as train_f:
        for line in tqdm(train_f):
            question = json.loads(line)
            question_text = question["question"]
            question_word = question_text.split()[0] if question_text.split()[0] in question_type_map else question_text.split()[1]
            answers = question['answers']
            answer_id, entity_id = {}, {}
            for answer in answers:
                answer_id.update(answer["kb_id"])
            for entity in entities:
                entity_id.update(entity['kb_id'])
            question_type[question_word].append(comRecall(answer_id, entity_id))
    

if __name__ == "__main__":
    # files = [train_file,dev_file,test_file]
    # for file in files:
    #     print(file)
    #     get_question_type(file)
    # f1_bound = 0.4
    # get_bad_question(0.4)
    # hybrid_bad = "/user_data/hybrid_bad"
    # get_bad_only()
    # get_bad_type()
    anaRecall()