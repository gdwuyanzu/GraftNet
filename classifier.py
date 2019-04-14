import torch
from torch.autograd import Variable
import torch.nn as nn
from util import use_cuda
import numpy as np

answer_type = 6
weight = use_cuda(Variable(torch.Tensor([1.26, 6.89, 23.66, 149.49, 104.40, 505.41])))

class Classifier(nn.Module):
    def __init__(self, pretrained_entity_emb_file, pretrained_entity_kge_file, num_entity, word_dim, kge_dim, entity_dim, linear_dropout):
        super(Classifier, self).__init__()
        self.has_entity_kge = True
        self.num_entity = num_entity
        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.kge_dim = kge_dim
        
        # initialize entity embedding
        self.entity_embedding =  nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=word_dim, padding_idx=num_entity)
        if pretrained_entity_emb_file is not None:
            self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_entity_emb_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False
        if pretrained_entity_kge_file is not None:
            self.has_entity_kge = True
            self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim, padding_idx=num_entity)
            self.entity_kge.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_entity_kge_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.entity_kge.weight.requires_grad = False

        # entity linear
        if self.has_entity_kge:
            self.entity_linear1 = nn.Linear(in_features=word_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear1 = nn.Linear(in_features=word_dim, out_features=entity_dim)
        self.entity_linear2 = nn.Linear(in_features=entity_dim, out_features=20)
        self.entity_linear3 = nn.Linear(in_features=20, out_features=answer_type)

        # dropout
        self.linear_drop = nn.Dropout(p=linear_dropout)
        # non-linear activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # loss
        self.bce_loss_logits = nn.BCEWithLogitsLoss()
        self.cross_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, batch):
        entities, answer_dist = batch

        # numpy to tensor
        entities = use_cuda(Variable(torch.from_numpy(entities).type('torch.LongTensor'), requires_grad=False))
        answer_dist = use_cuda(Variable(torch.from_numpy(answer_dist).type('torch.LongTensor'), requires_grad=False))

        # entity embedding
        entity_emb = self.entity_embedding(entities)
        if self.has_entity_kge:
            entity_emb = torch.cat((entity_emb, self.entity_kge(entities)), dim=2) # batch_size, max_local_entity, word_dim + kge_dim
        if self.word_dim != self.entity_dim:
            entity_emb = self.entity_linear1(self.linear_drop(entity_emb)) # batch_size, max_local_entity, entity_dim
        entity_emb = self.relu(self.entity_linear2(self.linear_drop(entity_emb)))
        score = self.relu(self.entity_linear3(self.linear_drop(entity_emb)))

        score = score.squeeze()
        loss = self.cross_loss(score, answer_dist)
        pred_dist = self.sigmoid(score)
        pred = torch.max(pred_dist,dim=1)[1]

        return loss, pred, pred_dist
        
        