import os
import os.path as path
import re
import sys
import numpy as np
import torch

def parser(f):
    triples = list()
    for i, triple in enumerate(f):
        # extract subject
        sub = triple.strip().replace("<", "").split(">")[0]
        sub = sub[sub.rfind("/")+1:]
        # extract content from "content"
        if "\"" in sub:
            pattern = re.compile('"(.*)"')
            try:
                sub_new = pattern.findall(sub)[0]
            except IndexError:
                # like "United States/Australian victory"
                sub = sub.replace("\"", "").strip()
                sub_new = sub
        # extract content from ":content"
        elif ":" in sub:
            pattern = re.compile(':(.*)')
            sub_new = pattern.findall(sub)[0]
        else:
            sub_new = sub
        sub_new = sub_new.replace(" ", "")

        # extract object
        obj = triple.strip().replace("<", "").split(">")[2]
        # fix extract content form "content\"
        if obj.rfind("/")+1 == len(obj):
            obj = obj[:-1]
        obj = obj[obj.rfind("/")+1:]
        # extract content from "content"
        if "\"" in obj:
            pattern = re.compile('"(.*)"')
            try:
                obj_new = pattern.findall(obj)[0]
            except IndexError:
                # like "United States/Australian victory"
                obj = obj.replace("\"", "").strip()
                obj_new = obj
        # extract content from ":content"
        elif ":" in obj:
            pattern = re.compile(':(.*)')
            obj_new = pattern.findall(obj)[0]
        else:
            obj_new = obj
        obj_new = obj_new.replace(" ", "")
        if obj_new == "":
            obj_new = "UNK"
        
        # extract predicate
        pred = triple.strip().replace("<", "").split(">")[1]
        pred = pred[pred.rfind("/")+1:]
        if "#" in pred:
            pattern = re.compile('#(.*)')
            pred_new = pattern.findall(pred)[0]
        elif ":" in pred:
            pattern = re.compile(':(.*)')
            pred_new = pattern.findall(pred)[0]
        else:
            pred_new = pred
        pred_new = pred_new.replace(" ", "")
        if not (sub_new == "" or pred_new == "" or obj_new == ""):
            triple_tuple = (i, sub, pred, obj, sub_new.replace(" ", ""), pred_new.replace(" ", ""), obj_new.replace(" ", ""))
            triples.append(triple_tuple)
        else:
            print(triple)
    return triples

# prepare data for per entity
def prepare_data(db_path, num):
    with open(path.join(db_path, 
        "{}".format(num), 
        "{}_desc.nt".format(num)),
        encoding="utf8") as f:
        triples = parser(f)
    return triples

# prepeare label for per label
def prepare_label(db_path, num, top_n, file_n):
    per_entity_label_dict = {}
    for i in range(file_n):
        with open(path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as f:
            labels  = parser(f)
            for _, _, _, _, _, pred_new, obj_new in labels:
                counter(per_entity_label_dict, "{}++$++{}".format(pred_new, obj_new))
    return per_entity_label_dict

# dict counter
def counter(cur_dict, word):
    if word in cur_dict:
        cur_dict[word] += 1
    else:
        cur_dict[word] = 1

def process_data(db_name, db_start, db_end, top_n=10, file_n=6):
    if db_name == "dbpedia":
        db_path = path.join(path.join(path.dirname(os.getcwd()), "data"), "dbpedia")
    elif db_name == "lmdb":
        db_path = path.join(path.join(path.dirname(os.getcwd()), "data"), "lmdb")
    else:
        raise ValueError("The database's name must be dbpedia or lmdb")

    data, data_for_transE = [], []
    label = []
    for i in range(db_start[0], db_end[0]):
        per_entity_data = prepare_data(db_path, i)
        # data: (sub, pred, obj)
        data.append([[pred_new, obj_new]for _, _, _, _, _, pred_new, obj_new in per_entity_data])
        # data_for_transE: (sub, obj, pred)
        data_for_transE.extend([[sub_new, obj_new, pred_new]for _, _, _, _, sub_new, pred_new, obj_new in per_entity_data])

    for i in range(db_start[1], db_end[1]):
        per_entity_data = prepare_data(db_path, i)
        data.append([[pred_new, obj_new]for _, _, _, _, _, pred_new, obj_new in per_entity_data])
        data_for_transE.extend([[sub_new, obj_new, pred_new]for _, _, _, _, sub_new, pred_new, obj_new in per_entity_data])

    for i in range(db_start[0], db_end[0]): 
        per_entity_label_dict = prepare_label(db_path, i, top_n=top_n, file_n=file_n)
        label.append(per_entity_label_dict)

    for i in range(db_start[1], db_end[1]): 
        per_entity_label_dict = prepare_label(db_path, i, top_n=top_n, file_n=file_n)
        label.append(per_entity_label_dict)
        
    # entity dict
    entity2ix = {}
    for sub_new, obj_new, _ in data_for_transE:
        if sub_new not in entity2ix:
            entity2ix[sub_new] = len(entity2ix)
        if obj_new not in entity2ix:
            entity2ix[obj_new] = len(entity2ix)

    # pred dict
    pred2ix = {}  
    for _, _, pred_new in data_for_transE:
        if pred_new not in pred2ix:
            pred2ix[pred_new] = len(pred2ix)

    return data, data_for_transE, label, entity2ix, pred2ix

def gen_data_transE(db_name, entity_to_ix, pred_to_ix, data_for_transE):
    # make dir
    if db_name == "dbpedia":
        directory = path.join(path.join(path.dirname(os.getcwd()), "data"), "dbpedia_transE")
    elif db_name == "lmdb":
        directory = path.join(path.join(path.dirname(os.getcwd()), "data"), "lmdb_transE")
    else:
        raise ValueError("The database's name must be dbpedia or lmdb")
    if not path.exists(directory):
        os.makedirs(directory)

    with open(path.join(directory, "entity2id.txt"), "w", encoding="utf-8") as f:
        dict_sorted =  sorted(entity_to_ix.items(), key = lambda x:x[1], reverse = False)
        f.write("{}\n".format(len(entity_to_ix)))
        for entity in dict_sorted:
            f.write("{}\t{}\n".format(entity[0], entity[1]))

    with open(path.join(directory, "relation2id.txt"), "w", encoding="utf-8") as f:
        dict_sorted =  sorted(pred_to_ix.items(), key = lambda x:x[1], reverse = False)
        f.write("{}\n".format(len(pred_to_ix)))
        for relation in dict_sorted:
            f.write("{}\t{}\n".format(relation[0], relation[1]))

    with open(path.join(directory, "train2id.txt"), "w", encoding="utf-8") as f:    
        # train2id 
        f.write("{}\n".format(len(data_for_transE)))
        for [sub, obj, pred] in data_for_transE:
            f.write("{}\t{}\t{}\n".format(entity_to_ix[sub], entity_to_ix[obj], pred_to_ix[pred]))

# load transE
def build_dict(f_path):
    word2ix = {}
    with open(f_path, "r", encoding="utf-8") as f:
        for _, pair in enumerate(f):
            try:
                temp = pair.strip().split("\t")
                word2ix[temp[0]] = int(temp[1])
            except:
                print(temp)
    return word2ix

def build_vec(word2ix, word_embedding):
    word2vec = {}
    for word in word2ix:
        word2vec[word] = word_embedding[int(word2ix[word])]
    return word2vec

def load_transE(db_name):
    if db_name == "dbpedia":
        directory = path.join(path.join(path.dirname(os.getcwd()), "data"), "dbpedia_transE")
    elif db_name == "lmdb":
        directory = path.join(path.join(path.dirname(os.getcwd()), "data"), "lmdb_transE")
    else:
        raise ValueError("The database's name must be dbpedia or lmdb")

    entity2ix = build_dict(path.join(directory, "entity2id.txt"))
    pred2ix = build_dict(path.join(directory, "relation2id.txt"))

    embedding = np.load(path.join(directory, "transE_vec.npz"))
    entity_embedding = embedding["ent_embedding"]
    pred_embedding = embedding["rel_embedding"]

    entity2vec = build_vec(entity2ix, entity_embedding)
    pred2vec = build_vec(pred2ix, pred_embedding)
    return entity2vec, pred2vec, entity2ix, pred2ix

def tensor_from_data(entity2vec, pred2ix, data):
    pred_list, obj_list = [], []
    for pred, obj in data:
        pred_list.append(pred2ix[pred])
        obj_list.append(entity2vec[obj])
    pred_tensor = torch.tensor(pred_list).view(-1, 1)
    obj_tensor = torch.tensor(obj_list).unsqueeze(1)
    return pred_tensor, obj_tensor

def tensor_from_weight(tensor_size, data, label):
    weight_tensor = torch.zeros(tensor_size)
    for label_word in label:
        order = -1
        for pred, obj in data:
            order += 1
            data_word = "{}++$++{}".format(pred, obj)
            if label_word == data_word:
                weight_tensor[order] += label[label_word]
                break
    return weight_tensor / torch.sum(weight_tensor)

# split data for cross validation
def split_data(base, num, data, label):
    start = num * base
    end = (num + 1) * base
    test_data = data[start:end]
    test_label = label[start:end]
    train_data, train_label = [], []
    for i, triples in enumerate(data):
        if i not in range(start, end):
            train_data.append(triples)
    for i, triples in enumerate(label):
        if i not in range(start, end):
            train_label.append(triples)
    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
   # dbpedia
   #  1 - 100, 141 - 165
   #data, data_for_transE, label, entity2ix, pred2ix = process_data("dbpedia", [1, 141], [101, 166])
   #gen_data_transE("dbpedia", entity2ix, pred2ix, data_for_transE)
   # lmdb
   # 101 - 140, 166 - 176
   #data, data_for_transE, label, entity2ix, pred2ix = process_data("lmdb", [101, 166], [141, 176])
   #gen_data_transE("lmdb", entity2ix, pred2ix, data_for_transE)
   None