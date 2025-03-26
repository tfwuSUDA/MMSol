import os
import pickle
import re
from sklearn.calibration import LabelEncoder
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from Bio import SeqIO
from transformers import AutoTokenizer

amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

PRETRAIN_MODEL_PATH = './models/Protein_LLM/esm2_t6_8M_UR50D'

class MMSol_Dataset(Dataset):       

    def __init__(self, file_path1, max_pad_len, edge_fea_path, node_fea_path, GO_fea_path):
        super().__init__()
        eps = 0.001
        self.pdb_dir = edge_fea_path
        self.max_pad_length = max_pad_len
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
        
        data = self.read_fasta_to_dataframe(file_path1)
        self.id = pd.Series(data['data_source'])
        self.feature = pd.Series(data['feature'])
        self.feature = torch.tensor(data['feature'].tolist(), dtype=torch.float32)
        
        with open(GO_fea_path, 'rb') as f:
            self.GO_fea = pickle.load(f)
        self.x = data['sequence'].values

        labels = data['label'].astype(np.float32)  # 将 label 转换为浮点数
        labels_tensor = torch.tensor(labels.values, dtype=torch.float32)
        self.softlabel = np.ones([len(data), 2], dtype=np.float32)*eps
        self.softlabel = torch.tensor(self.softlabel, dtype=torch.float32)

        le = LabelEncoder()
        y_noise = le.fit_transform(data['label_noise'])
        y_noise = torch.tensor(y_noise, dtype=torch.float32)
        self.y = labels_tensor
        self.y_noise = y_noise
        
        with open(edge_fea_path, 'rb') as f:
            self.dismatrix = pickle.load(f)

        with open(node_fea_path, 'rb') as f:
            self.node_fea = pickle.load(f)


    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        
        id = self.id[idx]
        sequence = self.x[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_pad_length, truncation=True, padding='max_length')
        label = self.y[idx]
        label_noise = self.y_noise[idx]
        feature = self.feature[idx]  
        softlabel = self.softlabel[idx]

        if id in self.GO_fea:
            GO_fea = torch.tensor(self.GO_fea[id], dtype=torch.float32)
        else:
            GO_fea = torch.zeros(144, dtype=torch.float32)
        
        try:
            sequence_feature = self.node_fea[id]  
        except:
            print(f"Warning: cannot find node feature for {id}")
            print(len(self.node_fea))
        sequence_graph = self.dismatrix[id]
    
        return id, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), feature, GO_fea, label, label_noise, sequence_feature, sequence_graph, softlabel, idx

    def read_fasta_to_dataframe(self, file_path1):
        records = SeqIO.to_dict(SeqIO.parse(file_path1, "fasta"))

        pattern = re.compile(r'label=(-?\d*\.?\d+) label_noise=(\d+) feature=\[([^\]]+)\] GO=\[([^\]]+)\]')

        data_source = []
        sequences = []
        label_tag = []
        label_noise = []
        feature = []
        GO=[]
        
        for label, record in records.items():
            data_source.append(label)
            sequences.append(str(record.seq))
            
            match = pattern.search(record.description)
            if match:
                label_tag.append(match.group(1))
                label_noise.append(match.group(2))
                feature.append([float(x) for x in match.group(3).split(',')])
                GO.append([int(x) for x in match.group(4).split(',')])
            else:
                print(f"Warning: cannot match label and label_noise in {label}")
                label_tag.append(None)
                label_noise.append(None)
                feature.append(None)
                GO.append(None)
                
        df = pd.DataFrame({
            'data_source': data_source,
            'sequence': sequences,
            'label': label_tag,
            'label_noise': label_noise, 
            'feature': feature,
            'GO': GO
        })
        
        return df
    
    def sample_mask(self, filter_func):
        self.data = [sample for sample in self.data if filter_func(sample)]

    def update_corrupted_label(self, noise_label):
        self.y[:] = noise_label[:]

    def update_corrupted_softlabel(self, noise_label):
        self.softlabel[:] = noise_label[:]
    
    def update_corrupted_label_noise(self, noise_label):
        self.y_noise[:] = noise_label[:]

    def get_data_labels(self):
        return self.y
    
    def get_data_labels_noise(self):
        return self.y_noise

    def get_data_softlabel(self):
        return self.softlabel

    def get_data_source(self):
        return self.id


def collate_fn(batch):
    id, input_ids, attention_mask, feature, GO_fea, label, label_noise, sequence_feature, sequence_graph, softlabel, idx = zip(*batch)
    
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    feature = torch.stack(feature)
    GO_fea = torch.stack(GO_fea)
    label = torch.stack(label)
    label_noise = torch.stack(label_noise)
    softlabel = torch.stack(softlabel)
    idx = torch.tensor(idx, dtype=torch.long)

    sequence_feature = [torch.from_numpy(np.array(d)) for d in sequence_feature]
    sequence_feature_mask = [torch.ones_like(d, dtype=torch.float32) for d in sequence_feature]  
    sequence_feature = torch.nn.utils.rnn.pad_sequence(sequence_feature, batch_first=True)  
    sequence_feature_mask = torch.nn.utils.rnn.pad_sequence(sequence_feature_mask, batch_first=True, padding_value=0)  

    max_length = max([d.shape[0] for d in sequence_graph])
    padded_sequence_graph = []
    mask_sequence_graph = []  
    for d in sequence_graph:
        padded_d = torch.zeros(max_length, max_length)
        mask_d = torch.zeros(max_length, max_length)  
        padded_d[:d.shape[0], :d.shape[1]] = torch.from_numpy(d)
        mask_d[:d.shape[0], :d.shape[1]] = 1  
        padded_sequence_graph.append(padded_d)
        mask_sequence_graph.append(mask_d)  
    sequence_graph = torch.stack(padded_sequence_graph)
    mask_sequence_graph = torch.stack(mask_sequence_graph)  
    return id, input_ids, attention_mask, feature, GO_fea, label, label_noise, sequence_feature, sequence_feature_mask, sequence_graph, mask_sequence_graph, softlabel, idx
