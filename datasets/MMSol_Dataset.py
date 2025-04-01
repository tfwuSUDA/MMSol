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

# path

amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Model parameters
NUMBER_EPOCHS = 25
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 1
NUM_CLASSES = 1

# GCN parameters
GCN_FEATURE_DIM = 91
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4

PRETRAIN_MODEL_PATH = './models/Protein_LLM/esm2_t6_8M_UR50D'


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def get_edge_index(pdbdir,EPSILON = 8.):
    pdb = open(pdbdir,"r")
    atom_coordinate = []
    i = 0
    for row in pdb.readlines():
        row = re.sub(r'(\.\d{3})', r'\1   ', row) 
        col = row.split()
        if col[0] == "MODEL" and col[1] != "1":
            break
        if col[0] == "ATOM":
            if col[2] == "CA":
                try:
                    atom_coordinate.append((float(col[6]),float(col[7]),float(col[8])))
                except Exception:
                    print(pdbdir)
                    break
    dismatrix = []
    for rowa in range(len(atom_coordinate)):
        tempdis = []
        for rowb in range(len(atom_coordinate)):
            atoma = (atom_coordinate[rowa][0],atom_coordinate[rowa][1],atom_coordinate[rowa][2])
            atomb = (atom_coordinate[rowb][0],atom_coordinate[rowb][1],atom_coordinate[rowb][2])
            dis = distance.euclidean(atoma,atomb)
            if dis == 0.:
                tempdis.append(0)
            elif dis <= EPSILON:
                tempdis.append(round(float(dis),3))
            elif dis > EPSILON:
                tempdis.append(0)
        dismatrix.append(tempdis)
    dismatrix = np.where(np.array(dismatrix), 1, 0)
    contactmatrix = coo_matrix(dismatrix)
    edge_index = torch.LongTensor(np.vstack((contactmatrix.row,contactmatrix.col)))
    return dismatrix, edge_index


def load_features(sequence):
    blosum = np.array([blosum_dict[amino] for amino in sequence])
    aaphy7 = np.array([aaphy7_dict[amino] for amino in sequence])
    feature_matrix = np.concatenate([blosum, aaphy7], axis=1)
    
    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load('./lib/edge_features' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix

def load_blosum():
    with open('./fea_data/BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result

def load_aaphy7():
    with open('./fea_data/aa_phy7', 'r') as f:
        result = dict()
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [float(i) for i in line[1:]]
    return result

blosum_dict = load_blosum()
aaphy7_dict = load_aaphy7()

class MMSol_Dataset(Dataset):       

    def __init__(self, file_path1, max_pad_len, edge_fea_path, node_fea_path, GO_fea_path):
        super().__init__()
        eps = 0.001
        self.blosum = load_blosum()
        self.pdb_dir = edge_fea_path
        self.max_pad_length = max_pad_len
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
        self.edge_fea_path = edge_fea_path
        self.node_fea_path = node_fea_path
        self.GO_fea_path = GO_fea_path

        data = self.read_fasta_to_dataframe(file_path1)
        self.id = pd.Series(data['data_source'])
        self.feature = pd.Series(data['feature'])
        self.feature = torch.tensor(data['feature'].tolist(), dtype=torch.float32)
        
        with open(GO_fea_path, 'rb') as f:
            self.GO_fea = pickle.load(f)

        self.x = data['sequence'].values

        le = LabelEncoder()
        y_le = le.fit_transform(data['label'])  
        self.softlabel = np.ones([len(data), 2], dtype=np.float32)*eps
        for i in range(len(data)):
            self.softlabel[i, y_le[i]] = 1 - eps
        self.softlabel = torch.tensor(self.softlabel, dtype=torch.float32)
        y_le = torch.tensor(y_le, dtype=torch.long)

        le = LabelEncoder()
        y_noise = le.fit_transform(data['label_noise'])
        y_noise = torch.tensor(y_noise, dtype=torch.float32)
        self.y = y_le
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

        sequence_feature = self.node_fea[id] 
        sequence_graph = self.dismatrix[id]
    
        return id, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), feature, GO_fea, label, label_noise, sequence_feature, sequence_graph, softlabel, idx
    

    def read_fasta_to_dataframe(self, file_path1):
        records = SeqIO.to_dict(SeqIO.parse(file_path1, "fasta"))

        pattern = re.compile(r'label=(\d+) label_noise=(\d+) feature=\[([^\]]+)\] GO=\[([^\]]+)\]')

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

class MMSol_Dataset_Subset(MMSol_Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
        self.max_pad_length = dataset.max_pad_length
        self.pdb_dir = dataset.pdb_dir
        self.tokenizer = dataset.tokenizer
        self.GO_fea = dataset.GO_fea
        self.feature = dataset.feature
        self.x = dataset.x
        self.softlabel = dataset.softlabel
        self.y = dataset.y
        self.y_noise = dataset.y_noise
        self.dismatrix = dataset.dismatrix
        self.node_fea = dataset.node_fea
        
        self.subset_data = {
            'id': dataset.id[self.indices],
            'x': dataset.x[self.indices],
            'sequence': dataset.x[self.indices],
            'y': dataset.y[self.indices],
            'y_noise': dataset.y_noise[self.indices],
            'feature': dataset.feature[self.indices],
            'softlabel': dataset.softlabel[self.indices],
            'GO_fea': dataset.GO_fea,
            'dismatrix': dataset.dismatrix,
            'node_fea': dataset.node_fea
        }
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def get_data_labels(self):
        return self.subset_data['y']

    def get_data_labels_noise(self):
        return self.subset_data['y_noise']

    def get_data_softlabel(self):
        return self.subset_data['softlabel']

    def get_data_source(self):
        return self.subset_data['x']

    def update_corrupted_softlabel(self, clean_softlabels):
        self.subset_data['softlabel'][:] = clean_softlabels

    def update_corrupted_label(self, clean_labels):
        self.subset_data['y'][:] = clean_labels

    def update_corrupted_label_noise(self, updated_noises):
        self.subset_data['y_noise'][:] = updated_noises

    





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

    sequence_feature = [torch.from_numpy(d) for d in sequence_feature]  
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
