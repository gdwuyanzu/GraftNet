import sys
import torch
from tqdm import tqdm
import numpy as np
import json

from typedata_loader import TypedataLoader
from classifier import Classifier
from util import use_cuda, save_model, load_model, load_dict

entity2id_file = "datasets/webqsp/full/entities.txt"
train_file = './datasets/webqsp/full/classfy/train.json'
dev_file = "./datasets/webqsp/full/classfy/dev.json"
test_file = "./datasets/webqsp/full/classfy/test.json"

pred_file = './datasets/webqsp/full/classfy/pred'
# load_model = None
load_model = './datasets/webqsp/full/classfy/model'
save_model = './datasets/webqsp/full/classfy/model'

entity_emb_file = './datasets/webqsp/full/entity_emb_100d.npy'
entity_kge_file = './datasets/webqsp/full/entity_kge_100d.npy'
word_dim = 100
kge_dim = 100
entity_dim = 50
linear_dropout = 0.2
learning_rate = 0.001
epoch = 1000
batch_size = 10
gradient_clip = 1

def cal_type_acc(pred, target):
    num_correct = 0.0
    num_total = 0.0
    for i, p in enumerate(pred):
        # if target[i][p] == 1.0:
        if p != 0 and target[i] == p:
            num_correct += 1
            num_total += 1
    return 0 if num_total == 0 else num_correct / num_total

def get_model(entity2id):
    my_model = use_cuda(Classifier(entity_emb_file,entity_kge_file, len(entity2id), word_dim, kge_dim, entity_dim, linear_dropout))
    if load_model is not None:
        print("loading model from", load_model)
        pretrained_model_states = torch.load(load_model)
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        del pretrained_model_states['cross_loss.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)
    return my_model

def inference(my_model, valid_data, entity2id, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc = [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    test_batch_size = 20
    if log_info:
        f_pred = open(pred_file, 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        acc  = cal_type_acc(pred, batch[-1])
        if log_info:
            out_put(pred_dist, id2entity, iteration * test_batch_size, valid_data, f_pred)
        eval_loss.append(loss.data[0])
        eval_acc.append(acc)
    
    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

def out_put(pred_dist, id2entity, start_id, dataloader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        p_dist_l = [float(prob) for prob in p_dist.data.cpu().numpy()]
        f_pred.write(json.dumps({id2entity[int(dataloader.entities[data_id][0])] : p_dist_l}) + '\n')
        

def train():
    print("training ...")

    #prepare data
    entity2id = load_dict(entity2id_file)

    train_data = TypedataLoader(train_file, entity2id)
    dev_data = TypedataLoader(dev_file, entity2id)
    test_data = TypedataLoader(test_file, entity2id)

    my_model = get_model(entity2id)
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    best_dev_acc = 0.0
    for i in range(epoch):
        try:
            print('epoch', i)
            my_model.train()
            train_loss, train_acc = [], []
            for iteration in tqdm(range(train_data.num_data // batch_size)):
                batch = train_data.get_batch(iteration, batch_size)
                loss, pred, _ = my_model(batch)
                pred = pred.data.cpu().numpy()
                acc = cal_type_acc(pred, batch[-1])
                train_loss.append(loss.data[0])
                train_acc.append(acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), gradient_clip)
                optimizer.step()
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('avg_training_acc', sum(train_acc) / len(train_acc))

            print("validating ...")
            eval_acc = inference(my_model, dev_data, entity2id)
            if eval_acc > best_dev_acc and save_model:
                print("saving model to", save_model)
                torch.save(my_model.state_dict(), save_model)
                best_dev_acc = eval_acc

        except KeyboardInterrupt:
            break

    # Test set evaluation
    print("evaluating on test")
    print('loading model from ...', test_file)
    my_model.load_state_dict(torch.load(save_model))
    test_acc = inference(my_model, test_data, entity2id, log_info=True)
    print("test_acc:", test_acc)
    return test_acc

def test():
    entity2id = load_dict(entity2id_file)

    test_data = TypedataLoader(test_file, entity2id)
    my_model = get_model(entity2id)
    test_acc = inference(my_model, test_data, entity2id,log_info=True)
    return test_acc

if __name__ == "__main__":
    if '--train' == sys.argv[1]:
        train()
    elif '--test' == sys.argv[1]:
        test()
    else:
        assert False, "--train or --test?"