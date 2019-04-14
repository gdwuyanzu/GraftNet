import json
from tqdm import tqdm

train_file = "./datasets/webqsp/full/train.json"
dev_file = "./datasets/webqsp/full/dev.json"
test_file = "./datasets/webqsp/full/test.json"

classfy_train_file = './datasets/webqsp/full/classfy/train.json'
classfy_dev_file = "./datasets/webqsp/full/classfy/dev.json"
classfy_test_file = "./datasets/webqsp/full/classfy/test.json"

num_type = 6
question_type = {"what": 0, "who": 1, "where": 2, "when": 3, "which": 4, "how": 5}
def getAnswerType(question):
    question_text = question["question"]
    question_word = question_text.split()[0] if question_text.split()[0] in question_type else question_text.split()[1]
    answer_type = question_type[question_word]
    return answer_type

def construct_classfy_data():
    train_type, dev_type, test_type = [0] * num_type, [0] * num_type, [0] * num_type
    with open(train_file) as train_f, open(dev_file) as dev_f, open(test_file) as test_f, \
        open(classfy_train_file, 'w') as cla_train_f, open(classfy_dev_file, 'w') as cla_dev_f, open(classfy_test_file, 'w') as cla_test_f:
        for line in tqdm(train_f):
            question = json.loads(line)
            answers = question['answers']
            for answer in answers:
                answer_type = getAnswerType(question)
                cla_train_f.write(json.dumps({answer['kb_id']: answer_type}) + '\n')
                train_type[answer_type] += 1
        for line in tqdm(dev_f):
            question = json.loads(line)
            answers = question['answers']
            for answer in answers:
                answer_type = getAnswerType(question)
                cla_dev_f.write(json.dumps({answer['kb_id']: answer_type}) + '\n')
                dev_type[answer_type] += 1
        for line in tqdm(test_f):
            question = json.loads(line)
            answers = question['answers']
            for answer in answers:
                answer_type = getAnswerType(question)
                cla_test_f.write(json.dumps({answer['kb_id']: answer_type}) + '\n')
                test_type[answer_type] += 1
        # total_type_f.write(json.dumps(train_type))
        print(train_type)
        print("train_total: " + str(sum(train_type)))
        print(dev_type)
        print("dev_total: " + str(sum(dev_type)))
        print(test_type)
        print("test_total: " + str(sum(test_type)))
    
if __name__ == "__main__":
    construct_classfy_data()