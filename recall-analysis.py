import json
from tqdm import tqdm

train_file = "./datasets/webqsp/full/train.json"
dev_file = "./datasets/webqsp/full/dev.json"
test_file = "./datasets/webqsp/full/test.json"

recall_train = "model/webqsp/ana_res/recall_train"
recall_test = "model/webqsp/ana_res/recall_test"

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
    with open(train_file) as train_f, open(recall_train, 'w') as rec_train_f:
        question_type = {"where":[], "what": [], "which": [], "how": [], "when": [], "who": []}
        for line in tqdm(train_f):
            question = json.loads(line)
            question_text = question["question"]
            question_word = question_text.split()[0] if question_text.split()[0] in question_type else question_text.split()[1]
            answers = question['answers']
            answer_id, entity_id = set(), set()
            for answer in answers:
                answer_id.add(answer["kb_id"])
            entities = question["subgraph"]["entities"]
            for entity in entities:
                entity_id.add(entity['kb_id'])
            question_type[question_word].append(comRecall(answer_id, entity_id))
        for k, v in question_type.items():
            question_type[k] = sum(question_type[k]) / len(question_type[k])
        rec_train_f.write(json.dumps(question_type) + '\n')

    with open(test_file) as test_f, open(recall_test, 'w') as rec_test_f:
        question_type = {"where":[], "what": [], "which": [], "how": [], "when": [], "who": []}
        for line in tqdm(test_f):
            question = json.loads(line)
            question_text = question["question"]
            question_word = question_text.split()[0] if question_text.split()[0] in question_type else question_text.split()[1]
            answers = question['answers']
            answer_id, entity_id = set(), set()
            for answer in answers:
                answer_id.add(answer["kb_id"])
            entities = question["subgraph"]["entities"]
            for entity in entities:
                entity_id.add(entity['kb_id'])
            question_type[question_word].append(comRecall(answer_id, entity_id))
        for k, v in question_type.items():
            question_type[k] = sum(question_type[k]) / len(question_type[k])
        rec_test_f.write(json.dumps(question_type) + '\n')

if __name__ == "__main__":
    anaRecall()