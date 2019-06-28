from __future__ import unicode_literals, print_function, division
import os
import os.path as path
import sys
import argparse
ROOTDIR = path.dirname(os.getcwd())
DATADIR = path.join(ROOTDIR, "data")
sys.path.append(DATADIR)
import utils
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model

def train(esa, data, label, criterion, optimizer, n_epoch, save_every, directory, device, clip, entity2vec, pred2ix, regularization):
    if not path.exists(directory):
        os.makedirs(directory)
    for epoch in range(n_epoch):
        total_loss = 0
        for i in range(len(data)):
            esa.zero_grad()
            pred_tensor, obj_tensor = utils.tensor_from_data(entity2vec, pred2ix, data[i])
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            weight_tensor = utils.tensor_from_weight(len(data[i]), data[i], label[i]).to(device)
            atten_weight = esa(input_tensor)

            # loss
            if regularization:
                loss = criterion(atten_weight.view(-1), weight_tensor.view(-1)).to(device) + \
                    torch.sum(torch.abs(atten_weight))
            else:
                loss = criterion(atten_weight.view(-1), weight_tensor.view(-1)).to(device)

            # clip gradient
            _ = nn.utils.clip_grad_norm_(esa.parameters(), clip)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        total_loss = total_loss/len(data)
        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": esa.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss
                }, path.join(directory, "checkpoint_epoch_{}.pt".format(epoch)))
        print("epoch: {}".format(epoch), total_loss)

def train_iter(db_name, base, data, label, pred2ix, pred2ix_size, entity2vec, \
            pred_embedding_dim, transe_dim, hidden_size, \
            criterion, clip, lr, \
            n_epoch, save_every, regularization, device):
    if regularization == True:
        print("use regularization in training")
    for i in range(5):
        train_data, train_label, _, _ = utils.split_data(base, i, data, label)
        esa = model.ESA(pred2ix_size, pred_embedding_dim, transe_dim, hidden_size, device)
        esa.to(device)
        optimizer = optim.Adam(esa.parameters(), lr=lr, amsgrad=False)
        directory = os.path.join(os.getcwd(), "checkpoint-{}-{}".format(db_name, i))
        train(esa, train_data, train_label, criterion, optimizer, n_epoch, save_every, \
            directory, device, clip, entity2vec, pred2ix, regularization)

def writer(DB_DIR, skip_i, directory, top_or_rank, output):
    with open(path.join(DB_DIR, 
            "{}".format(skip_i+1), 
            "{}_desc.nt".format(skip_i+1)),
            encoding="utf8") as fin, \
    open(path.join(directory,
            "{}".format(skip_i+1),
            "{}_{}.nt".format(skip_i+1, top_or_rank)),
            "w", encoding="utf8") as fout:
        if top_or_rank == "top5" or top_or_rank == "top10":
            top_list = output.squeeze(0).numpy().tolist()
            for t_num, triple in enumerate(fin):
                if t_num in top_list:
                    fout.write(triple)
        elif top_or_rank == "rank":
            rank_list = output.squeeze(0).numpy().tolist()
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                    fout.write(triples[rank])
    return

def generator(DB_NAME, base, data, label, entity2vec, pred2ix, pred2ix_size, \
        pred_embedding_dim, transE_dim, hidden_size, devide, use_epoch, \
        ):
    directory = path.join(os.getcwd(), DB_NAME)
    if not path.exists(directory):
        os.makedirs(directory)

    print("generating entity summarization results:") 
    for num in tqdm(range(5)):
        CHECK_DIR = path.join(os.getcwd(), "checkpoint-{}-{}".format(DB_NAME, num))
        esa = model.ESA(pred2ix_size, args.pred_embedding_dim, args.transE_dim, hidden_size, device)
        checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch)))
        esa.load_state_dict(checkpoint["model_state_dict"])
        esa.to(device)
        for i in range(num*base, (num+1)*base):
            data_i = i - num*base
            _, _, test_data, test_label = utils.split_data(base, num, data, label)
            pred_tensor, obj_tensor = utils.tensor_from_data(entity2vec, pred2ix, test_data[data_i])
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            weight_tensor = utils.tensor_from_weight(len(test_data[data_i]), test_data[data_i], test_label[data_i]).to(device)
            atten_weight = esa(input_tensor)
            atten_weight = atten_weight.view(1, -1).cpu()
            weight_tensor = weight_tensor.view(1, -1).cpu()
            (_, label_top10) = torch.topk(weight_tensor, 10)
            (_, output_top10) = torch.topk(atten_weight, 10)
            (_, label_top5) = torch.topk(weight_tensor, 5)
            (_, output_top5) = torch.topk(atten_weight, 5)
            (_, output_rank) = torch.topk(atten_weight, len(test_data[data_i]))
            if num == 4:
                skip_i = i + skip_num + db_base
            else:
                skip_i = i + db_base
            if not path.exists(path.join(directory, "{}".format(skip_i+1))):
                os.makedirs(path.join(directory, "{}".format(skip_i+1)))
            writer(DB_DIR, skip_i, directory, "top10", output_top10)
            writer(DB_DIR, skip_i, directory, "top5", output_top5)
            writer(DB_DIR, skip_i, directory, "rank", output_rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESA: Entity Smmarization with Attention')
    parser.add_argument("--DB_NAME", type=str, default="dbpedia", help="use dbpedia or lmdb")
    parser.add_argument("--mode", type=str, default="all", help="train, test or all")
    parser.add_argument("--top_n", type=int, default=10, help="use top 5 or 10 gold(label) files")
    parser.add_argument("--file_n", type=int, default=6, help="the number of gold(label) files in ESBM benchmark")
    parser.add_argument("--transE_dim", type=int, default=100, help="the dimension of pretrained transE vectors")
    parser.add_argument("--pred_embedding_dim", type=int, default=100, help="the embeddiing dimension of predicate")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--clip", type=float, default=50.0, help="gradient clip")
    parser.add_argument("--save_every", type=int, default=4, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    parser.add_argument("--use_epoch", type=int, default=24, help="use which epoch to generate final summarization")
    parser.add_argument("--loss_function", type=str, default="BCE", help="use which loss function to train the model, BCE or MSE")
    parser.add_argument("--regularization", type=bool, default=False, help="use regularization or not")

    args = parser.parse_args()

    if args.DB_NAME == "dbpedia":
        DB_START, DB_END = [1, 141], [101, 166]
        base = 25
        skip_num = 40
        db_base = 0
    elif args.DB_NAME == "lmdb":
        DB_START, DB_END = [101, 166], [141, 176]
        base = 10
        skip_num = 25
        db_base = 100
    DB_DIR = path.join(DATADIR, args.DB_NAME)

    # load data
    data, _, label, _, _ = utils.process_data(args.DB_NAME, DB_START, DB_END, args.top_n, args.file_n)
    entity2vec, pred2vec, entity2ix, pred2ix = utils.load_transE(args.DB_NAME)
    pred2ix_size = len(pred2ix)
    hidden_size = args.transE_dim + args.pred_embedding_dim

    # train
    ## cuda 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda or cpu: {}".format(device))

    ## loss function
    if args.loss_function == "BCE":
        criterion = torch.nn.BCELoss()
    elif args.loss_function == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        print("please choose choose the correct loss fucntion")
        sys.exit()
    print("loss function: {}".format(args.loss_function))

    if args.mode == "train" or args.mode == "all":
        ## training iteration (5-fold cross validation)
        train_iter(args.DB_NAME, base, data, label, pred2ix, pred2ix_size, entity2vec, \
                args.pred_embedding_dim, args.transE_dim, hidden_size, \
                criterion, args.clip, args.lr, args.n_epoch, args.save_every, args.regularization, device)

    if args.mode == "test" or args.mode == "all":
    #generate
        generator(args.DB_NAME, base, data, label, entity2vec, pred2ix, pred2ix_size, \
                args.pred_embedding_dim, args.transE_dim, hidden_size, device, args.use_epoch, \
                )