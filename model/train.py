from __future__ import unicode_literals, print_function, division
import os
import os.path as path
import sys
ROOTDIR = path.dirname(os.getcwd())
DATADIR = path.join(ROOTDIR, "data")
sys.path.append(DATADIR)
import utils

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model

# cuda 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# database config
DB_NAME = "dbpedia"
#DB_NAME = "lmdb"
DB_DIR = path.join(DATADIR, DB_NAME)
DB_START, DB_END = [1, 141], [101, 166]
#DB_START, DB_END = [101, 166], [141, 176]
top_n, file_n = 10, 6

# load data and dict from transE
data, _, label, _, _ = utils.process_data(DB_NAME, DB_START, DB_END, top_n, file_n)
entity2vec, pred2vec, entity2ix, pred2ix = utils.load_transE(DB_NAME)

# model config
transE_dim = 100
pred_embedding_dim = 100
pred2ix_size = len(pred2ix)
hidden_size = transE_dim + pred_embedding_dim
clip = 50.0
# train config
save_every = 5
n_epoch = 50
epoch = 36

def train(es_attention, data, label, criterion, optimizer, n_epoch, save_every, directory, device):
    if not path.exists(directory):
        os.makedirs(directory)
    for epoch in range(n_epoch):
        total_loss = 0
        for i in range(len(data)):
            es_attention.zero_grad()
            pred_tensor, obj_tensor = utils.tensor_from_data(entity2vec, pred2ix, data[i])
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            weight_tensor = utils.tensor_from_weight(len(data[i]), data[i], label[i]).to(device)
            atten_weight = es_attention(input_tensor)

            # loss
            loss = criterion(atten_weight.view(-1), weight_tensor.view(-1)).to(device)
            #loss = criterion(atten_weight.view(-1), weight_tensor.view(-1)).to(device) + \
            #   torch.sum(torch.abs(atten_weight))

            # clip gradient
            _ = nn.utils.clip_grad_norm_(es_attention.parameters(), clip)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        total_loss = total_loss/len(data)
        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": es_attention.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss
                }, path.join(directory, "checkpoint_epoch_{}.pt".format(epoch)))
        print("epoch: {}".format(epoch), total_loss)

if __name__ == "__main__":
    if DB_NAME == "dbpedia":
        base = 25
        skip_num = 40
        db_base = 0
    elif DB_NAME == "lmdb":
        base = 10
        skip_num = 25
        db_base = 100
    for i in range(5):
        train_data, train_label, _, _ = utils.split_data(base, i, data, label)
        es_attention = model.ES_Attention(pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, device)
        es_attention.to(device)
        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(es_attention.parameters(), lr=0.0001, amsgrad=False)
        directory = os.path.join(os.getcwd(), "checkpoint-{}-{}".format(DB_NAME, i))
        train(es_attention, train_data, train_label, criterion, optimizer, n_epoch, save_every, directory, device)

    #eval
    directory = path.join(os.getcwd(), DB_NAME)
    if not path.exists(directory):
        os.makedirs(directory)
    
    for num in range(5):
        CHECK_DIR = path.join(os.getcwd(), "checkpoint-{}-{}".format(DB_NAME, num))
        es_attention = model.ES_Attention(pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, device)
        checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(epoch)))
        es_attention.load_state_dict(checkpoint["model_state_dict"])
        es_attention.to(device)
        for i in range(num*base, (num+1)*base):
            data_i = i - num*base
            _, _, test_data, test_label = utils.split_data(base, num, data, label)
            pred_tensor, obj_tensor = utils.tensor_from_data(entity2vec, pred2ix, test_data[data_i])
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            weight_tensor = utils.tensor_from_weight(len(test_data[data_i]), test_data[data_i], test_label[data_i]).to(device)
            atten_weight = es_attention(input_tensor)
            atten_weight = atten_weight.view(1, -1).cpu()
            weight_tensor = weight_tensor.view(1, -1).cpu()
            (_, label_top10) = torch.topk(weight_tensor, 10)
            (_, output_top10) = torch.topk(atten_weight, 10)
            (_, label_top5) = torch.topk(weight_tensor, 5)
            (_, output_top5) = torch.topk(atten_weight, 5)
            (_, output_rank) = torch.topk(atten_weight, len(test_data[data_i]))
            # top10
            if num == 4:
                skip_i = i + skip_num + db_base
            else:
                skip_i = i + db_base
            if not path.exists(path.join(directory, "{}".format(skip_i+1))):
                os.makedirs(path.join(directory, "{}".format(skip_i+1)))
            with open(path.join(DB_DIR, 
                    "{}".format(skip_i+1), 
                    "{}_desc.nt".format(skip_i+1)),
                    encoding="utf8") as fin, \
                open(path.join(directory,
                    "{}".format(skip_i+1),
                    "{}_top{}.nt".format(skip_i+1, 10)),
                    "w", encoding="utf8") as fout:
                top_list = output_top10.squeeze(0).numpy().tolist()
                for t_num, triple in enumerate(fin):
                    if t_num in top_list:
                        fout.write(triple)
            # top5 
            with open(path.join(DB_DIR, 
                    "{}".format(skip_i+1), 
                    "{}_desc.nt".format(skip_i+1)),
                    encoding="utf8") as fin, \
                open(path.join(directory,
                    "{}".format(skip_i+1),
                    "{}_top{}.nt".format(skip_i+1, 5)),
                    "w", encoding="utf8") as fout:
                top5_list = output_top5.squeeze(0).numpy().tolist()
                for t_num, triple in enumerate(fin):
                    if t_num in top5_list:
                        fout.write(triple)
            # rank 
            with open(path.join(DB_DIR, 
                    "{}".format(skip_i+1), 
                    "{}_desc.nt".format(skip_i+1)),
                    encoding="utf8") as fin, \
                open(path.join(directory,
                    "{}".format(skip_i+1),
                    "{}_rank.nt".format(skip_i+1)),
                    "w", encoding="utf8") as fout:
                rank_list = output_rank.squeeze(0).numpy().tolist()
                triples = [triple for _, triple in enumerate(fin)]
                for rank in rank_list:
                    fout.write(triples[rank])