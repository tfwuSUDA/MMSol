import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve

from datasets.MMSol_Dataset import collate_fn, MMSol_Dataset
from utils.SparseGO.utils_conform import *
from models.MMSol import Model
from configs import config_noise

from tqdm import tqdm
import argparse

import os
import torch.nn.parallel
import random
from termcolor import cprint

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser(description='Training for MMSol Noise')

parser.add_argument('--epochs', default=config_noise.epochs, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=config_noise.batch_size, type=int, help='Batch size')
parser.add_argument('--lr', default=config_noise.lr, type=float, help='Learning rate')
parser.add_argument('--seed', default=config_noise.seed, type=int, help='Random seed')
parser.add_argument('--num_workers', default=config_noise.num_workers, type=int, help='Number of workers for data loading')
parser.add_argument('--weight_decay', default=config_noise.weight_decay, type=float, help='Weight decay for optimizer')
parser.add_argument('--gpu', default=config_noise.gpu, type=int, help='GPU number')
parser.add_argument('--train_dataset_path', default=config_noise.train_dataset_path, type=str, help='Path for train dataset')
parser.add_argument('--valid_dataset_path', default=config_noise.valid_dataset_path, type=str, help='Path for valid dataset')
parser.add_argument('--max_pad_len', default=config_noise.max_pad_len, type=int, help='Max pad length for sequence')
parser.add_argument('--model_path', default=config_noise.model_path, type=str, help='Path for model')
parser.add_argument('--save_path', default=config_noise.save_path, type=str, help='Path for save the best model')
parser.add_argument('--train_dataset_changed_path', default=config_noise.train_dataset_changed_path, type=str, help='train_dataset_changed_path')
parser.add_argument('--protein2id', default=config_noise.protein2id, type=str, help='protein2id')
parser.add_argument('--protein2ont', default=config_noise.protein2ont, type=str, help='protein2ont')

args = parser.parse_args()


def updateA(s, h, rho=0.9):
    
    eps = 0.1  
    h = torch.tensor(h, dtype=torch.float32).reshape(-1, 1)  
    s = s.clone().detach().requires_grad_(False).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps 
    A[s.argmax(0)] = rho   

    result = -((A.matmul(s)).t()).matmul(h) 

    return result, A

def initA(s, rho=0.9):
    
    eps = 0.1  
    s = s.clone().detach().requires_grad_(False).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps  
    A[s.argmax(0)] = rho   

    return A

def lrt_flip_scheme(pred_softlabels_bar, y_tilde, y_noise, delta1, delta2):
    '''
    Label changed
    If a sample with label_noise marked as 2 is modified, label_noise will be marked as 4
    If a sample with label_noise marked as 1 is modified, label_noise will be marked as 5
    '''
    ntrain = pred_softlabels_bar.shape[0]  
    num_class = pred_softlabels_bar.shape[1]  
    
    total_updates = 0
    updated_indices = []

    for i in range(ntrain):
        cond_1 = (pred_softlabels_bar[i].max()/pred_softlabels_bar[i][y_tilde[i]] > delta1)  
        cond_2 = (pred_softlabels_bar[i].max()/pred_softlabels_bar[i][y_tilde[i]] > delta2)
        cond_3 = (y_noise[i] == 1)  
        cond_4 = (y_noise[i] == 2)

        if cond_1 and cond_4:  
            y_tilde[i] = pred_softlabels_bar[i].argmax()  
            y_noise[i] = 4  
            total_updates += 1
            updated_indices.append(i)
        elif cond_2 and cond_3:  
            y_tilde[i] = pred_softlabels_bar[i].argmax()  
            y_noise[i] = 5  
            total_updates += 1
            updated_indices.append(i)

    eps = 0.001
    clean_softlabels = torch.ones(ntrain, num_class)*eps/(num_class - 1)
    y_tilde_tensor = torch.tensor(np.array(y_tilde), dtype=torch.long).to(clean_softlabels.device)
    clean_softlabels.scatter_(1, y_tilde_tensor.reshape(-1, 1), 1 - eps)

    print(f"Total updates: {total_updates}")

    return y_noise, y_tilde, clean_softlabels, updated_indices, total_updates


def read_fasta_to_dataframe(file_path1):
        records = SeqIO.to_dict(SeqIO.parse(file_path1, "fasta"))

        pattern = re.compile(r'label=(\d+) label_noise=(\d+) feature=\[([^\]]+)\] GO=\[([^\]]+)\]')

        data_source = []
        sequences = []
        label_tag = []
        label_noise = []
        feature = []
        GO=[]
        
        for id, record in records.items():
            data_source.append(id)
            sequences.append(str(record.seq))
            
            match = pattern.search(record.description)
            if match:
                label_tag.append(match.group(1))
                label_noise.append(match.group(2))
                feature.append([float(x) for x in match.group(3).split(',')])
                GO.append([int(x) for x in match.group(4).split(',')])
            else:
                print(f"Warning: cannot match label and label_noise in {id}")
                label_tag.append(None)
                label_noise.append(None)
                feature.append(None)
                GO.append(None)
                
        # 创建 DataFrame
        df = pd.DataFrame({
            'data_source': data_source,
            'sequence': sequences,
            'label': label_tag,
            'label_noise': label_noise, 
            'feature': feature,
            'GO': GO
        })
        
        return df

def reconstruct_fasta(df, train_dataset, output_file):
    '''
    Fasta file reconstruction 
    Updated to the modified dataset
    '''
    id_to_label = {}
    id_to_label_noise = {}
    for idx in range(len(train_dataset)):
        id_to_label[train_dataset[idx][0]] = int(train_dataset[idx][5])  
        id_to_label_noise[train_dataset[idx][0]] = int(train_dataset[idx][6])  

    sequences_new = []
    for _, raw in df.iterrows():
        id = raw['data_source']

        new_label = int(id_to_label[id])  
        new_label_noise = int(id_to_label_noise[id])  

        new_description = 'label={} label_noise={} feature={} GO={}'.format(new_label, new_label_noise, raw['feature'], raw['GO'])

        seq_new = SeqRecord(Seq(raw['sequence']), id=id, description=new_description, name='')
        sequences_new.append(seq_new)

    with open(output_file, "w") as output_handle:
        SeqIO.write(sequences_new, output_handle, "fasta")
        

class Ecoli(MMSol_Dataset):

    def __init__(self):
        global args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.train_dataset  = args.train_dataset_path
        self.train_changed_dataset = args.train_dataset_changed_path
        self.valid_dataset = args.valid_dataset_path
        self.model_path = args.model_path
        self.seed = args.seed
        self.save_path = args.save_path
        self.num_workers = args.num_workers  
        self.weight_decay = args.weight_decay
        self.max_pad_len = args.max_pad_len
        self.epoch_start = 2 
        self.every_n_epoch = 1  
        self.epoch_update = 2     
        self.epoch_interval = 1
        self.train_val_ratio = 0.9
        self.protein2id = args.protein2id
        self.protein2ont = args.protein2ont

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def train(self):
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")
        
        # -----SparceGO-----
        protein2id_mapping = load_mapping(self.protein2id)

        dG, terms_pairs, proteins_terms_pairs = load_ontology(self.protein2ont, protein2id_mapping)

        sorted_pairs, level_list, level_number = sort_pairs(
            proteins_terms_pairs, terms_pairs, dG, protein2id_mapping)

        layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)  
        
        # -----Model----- 
        model = Model(layer_connections=layer_connections)
        # print(model)

        model = model.to(device)
        model.train()

        # -----Dataset-----
        train_dataset = MMSol_Dataset(self.train_dataset, max_pad_len=self.max_pad_len, 
                                      edge_fea_path='./data/noise/noise_graph/train_LPE_5_1.pkl', 
                                      node_fea_path='./data/noise/noise_graph/train_node.pkl', 
                                      GO_fea_path='./data/noise/noise_go/train_go.pkl')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                  shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        ntrain = len(train_dataset)
        
        # -----Loss-----
        num_class = 2  

        A = 1/num_class*torch.ones(ntrain, num_class, num_class, requires_grad=False).float().to(device)  
        h = np.zeros([ntrain, num_class])  
        
        total_labels = train_dataset.get_data_labels()  
        total_samples = len(total_labels)
        num_pos_samples = (total_labels == 1).sum().item() 
        num_neg_samples = (total_labels == 0).sum().item()
        
        pos_weight = total_samples / (2 * num_pos_samples)  
        neg_weight = total_samples / (2 * num_neg_samples) 
        print(f'pos_weight: {pos_weight}, neg_weight: {neg_weight}')
        weights = torch.tensor([neg_weight, pos_weight]).to(device)
        criterion_1 = nn.NLLLoss(weight=weights).to(device)  
        
        pred_softlabels = np.zeros([ntrain, self.every_n_epoch, num_class], dtype=float)  
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)  

        best_loss = np.inf
        best_loss_model = None

        for epoch in range(1,self.epochs+1): 
            model.train()
            
            train_loss = 0
            corrects_epoch = 0  
            total_epoch = 0  

            delta1 = 1.2 + 0.02*max(epoch - self.epoch_update + 1, 0)
            delta2 = 3.5 + 0.02*max(epoch - self.epoch_update + 1, 0)

            loop = tqdm(total=len(train_loader), leave=False)

            if epoch == self.epoch_start + self.epoch_interval + 1:
                print("-----Update Dataset-----")

                raw_df = read_fasta_to_dataframe(self.train_dataset)  
  
                reconstruct_fasta(raw_df, train_dataset, self.train_changed_dataset)  
                
                del train_dataset
                del train_loader
                import gc
                gc.collect()
                
                train_dataset_new = MMSol_Dataset(self.train_changed_dataset, max_pad_len=self.max_pad_len, 
                                                  edge_fea_path='./data/noise/noise_graph/train_LPE_5_1.pkl', 
                                                  node_fea_path='./data/noise/noise_graph/train_node.pkl', 
                                                  GO_fea_path='./data/noise/noise_go/train_go.pkl')
                train_loader = DataLoader(train_dataset_new, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
                ntrain_new = len(train_dataset_new)
                optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)
                A_new = 1/num_class*torch.ones(ntrain_new, num_class, num_class, requires_grad=False).float().to(device)
                loop = tqdm(total=len(train_loader), leave=False)
                
                total_labels_new = train_dataset_new.get_data_labels()  
                total_samples_new = len(total_labels_new)
                num_pos_samples_new = (total_labels_new == 1).sum().item()
                num_neg_samples_new = (total_labels_new == 0).sum().item()
                
                pos_weight_new = total_samples_new / (2 * num_pos_samples_new) 
                neg_weight_new = total_samples_new / (2 * num_neg_samples_new) 
                
                weights_2 = torch.tensor([neg_weight_new, pos_weight_new]).to(device)
                criterion_2 = nn.NLLLoss(weight=weights_2).to(device)  
                
                y_soft_new = train_dataset_new.get_data_softlabel()  
                with torch.no_grad():
                    for i in tqdm(range(ntrain_new), ncols=100, ascii=True):  
                        try:
                            A_opt_new = initA(y_soft_new[i], rho=0.9)
                        except:
                            A_new[i] = A_new[i]
                            unsolved += 1
                            continue
                        A_new[i] = A_opt_new.clone().detach().requires_grad_(False)
                print("-----Update Dataset Done-----")

            for batch in train_loader:
                
                id, inputs, attention_ids, feature, GO_fea, labels_cpu, label_noise, sequences_feature, sequences_mask, graph_feature, graph_mask, softlabels, index = batch
                    
                inputs = inputs.to(device)
                attention_ids = attention_ids.to(device)
                feature = feature.to(device)
                GO_fea = GO_fea.to(device)
                labels = labels_cpu.to(device)
                sequences_feature = sequences_feature.to(device)
                sequences_mask = sequences_mask.to(device)
                graph_feature = graph_feature.to(device)
                graph_mask = graph_mask.to(device)
                softlabels = softlabels.to(device)
                outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask,  graph_mask)
                labels = labels.long()
                log_outputs = torch.log_softmax(outputs, 1).float()

                if epoch in [self.epoch_start, self.epoch_start+self.epoch_interval]:
                    index_1d = np.asarray(index).flatten()
                    h[index_1d, :] = log_outputs.detach().cpu()  
                normal_outputs = torch.softmax(outputs, 1)  

                if epoch < self.epoch_start:
                    loss = criterion_1(log_outputs, labels)
                elif epoch >= self.epoch_start and epoch <= self.epoch_start + self.epoch_interval: 
                    A_batch = A[index].to(device)  
                    loss = 0.5*sum([-A_batch[i].matmul(softlabels[i].reshape(-1, 1).float()).t().matmul(log_outputs[i])
                                for i in range(len(index))]) / len(index) + \
                        0.5*criterion_1(log_outputs, labels)
                else:
                    loss = criterion_2(log_outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= (self.epoch_update - self.every_n_epoch) and epoch <= self.epoch_start + self.epoch_interval:
                    pred_softlabels[index, epoch % self.every_n_epoch, :] = normal_outputs.detach().cpu().numpy()  

                preds_noise = torch.argmax(outputs, dim=1)
                corrects = (preds_noise == labels).float().sum()  
                corrects_epoch += corrects.item()
                total_epoch += inputs.size(0)
                train_loss += loss.item()
                
                loop.set_description(f"f'Epoch {epoch} | Loss {loss.item():.5f}'")
                loop.set_postfix(loss=loss.item(), acc=corrects.item()/inputs.size(0))   
                if loss.item() == np.nan:
                    print('nan')
                    break
                loop.update() 

            train_acc = corrects_epoch / total_epoch
            print(f"Train Accuracy: {train_acc:.5f}")
            print(f'Epoch {epoch}, Total Training Loss: {train_loss}')

            valid_acc, valid_f1, valid_auc, valid_precision, valid_recall, valid_mcc, val_loss = self.valid(model=model)
            metrics_path = os.path.join(self.save_path, f"log.txt")
            with open(metrics_path, 'a') as f:
                f.write(f"Epoch: {epoch} Valid ACC: {valid_acc:.5f} Valid F1: {valid_f1:.5f} Valid AUC: {valid_auc} Valid Precision: {valid_precision} Valid Recall: {valid_recall} Valid MCC: {valid_mcc}\n")

            # Save the model 
            if val_loss < best_loss:
                best_loss = val_loss 
                best_loss_model = model.state_dict()

                model_path = os.path.join(self.save_path, f"best_loss_model.pth")
                torch.save(best_loss_model, model_path)
                print(f"Best Loss model saved.")

                metrics_path = os.path.join(self.save_path, f"best_loss_model.txt")
                with open(metrics_path, 'w') as f:
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"Valid ACC: {valid_acc:.5f}\n")
                    f.write(f"Valid AUC: {valid_auc:.5f}\n")
                    f.write(f"Valid MCC: {valid_mcc:.5f}\n")
                    f.write(f"Valid F1: {valid_f1:.5f}\n")
                    f.write(f"Valid Precision: {valid_precision:.5f}\n")
                    f.write(f"Valid Recall: {valid_recall:.5f}\n")

            if epoch in [self.epoch_start, self.epoch_start + self.epoch_interval]:  
                cprint("+++++++++++++++++ Updating A +++++++++++++++++++", "magenta")
                unsolved = 0
                infeasible = 0
                y_soft = train_dataset.get_data_softlabel()  
                sample_ids = train_dataset.get_data_source()

                with torch.no_grad():
                    for i in tqdm(range(ntrain), ncols=100, ascii=True): 
                        try:
                            result, A_opt = updateA(y_soft[i], h[i], rho=0.9)  
                        except:
                            A[i] = A[i]
                            unsolved += 1
                            continue

                        if (result == np.inf):
                            A[i] = A[i]
                            infeasible += 1
                        else:
                            A[i] = A_opt.clone().detach().requires_grad_(False)
                print("Unsolved points: {} | Infeasible points: {}".format(unsolved, infeasible))

            if epoch >= self.epoch_update and epoch <= self.epoch_start + self.epoch_interval:  
                '''
                label changed
                '''
                y_tilde = train_dataset.get_data_labels() 
                y_noise = train_dataset.get_data_labels_noise() 
                pred_softlabels_bar = pred_softlabels.mean(1) 
                updated_noises, clean_labels, clean_softlabels, updated_indices, updates = lrt_flip_scheme(pred_softlabels_bar, y_tilde, y_noise, delta1, delta2)  
                train_dataset.update_corrupted_softlabel(clean_softlabels)  
                train_dataset.update_corrupted_label(clean_softlabels.argmax(1))  
                train_dataset.update_corrupted_label_noise(updated_noises)  

            
            
                
    def valid(self, model=None):         
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")
        
        model.eval()
        
        # -----dataset-----

        val_dataset = MMSol_Dataset(self.valid_dataset, max_pad_len=self.max_pad_len, 
                                    edge_fea_path='./data/noise/noise_graph/valid_LPE_5_1.pkl', 
                                    node_fea_path='./data/noise/noise_graph/valid_node.pkl', 
                                    GO_fea_path='./data/noise/noise_go/valid_go.pkl')
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, 
                                num_workers=self.num_workers, collate_fn=collate_fn)

        criterion_1 = nn.NLLLoss().to(device)  
        val_loss = 0

        predicted_list = []
        labels_list = []
        prob_all = []

        loop = tqdm(val_loader)
        with torch.no_grad():
            for i, data in enumerate(loop, 0):
                
                id, inputs, attention_ids, feature, GO_fea, labels_cpu, label_noise, sequences_feature, sequences_mask, graph_feature, graph_mask, _, _ = data
                    
                inputs = inputs.to(device)
                attention_ids = attention_ids.to(device)
                feature = feature.to(device)
                GO_fea = GO_fea.to(device)
                labels = labels_cpu.to(device)
                sequences_feature = sequences_feature.to(device)
                sequences_mask = sequences_mask.to(device)
                graph_feature = graph_feature.to(device)
                graph_mask = graph_mask.to(device)

                outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask,  graph_mask)
                
                labels = labels.long()
                log_outputs = torch.log_softmax(outputs, 1).float()
                loss = criterion_1(log_outputs, labels)
            
                outputs = F.softmax(outputs, dim=1)
                prob_all.extend(outputs[:,1].cpu().numpy()) 
                predicted = torch.argmax(outputs, dim=1)
                predicted_list += predicted.tolist()
                
                labels_list += labels.tolist()
                val_loss += loss.item()
                
        predicted_array = np.array(predicted_list)
        labels_array = np.array(labels_list)
        prob_all = np.array(prob_all)

        test_auc = roc_auc_score(labels_array, prob_all)

        fpr, tpr, thresholds = roc_curve(labels_array, prob_all)

        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_idx]

        p = prob_all.copy()
        p[p>=best_threshold] = 1
        p[p<best_threshold] = 0

        acc = accuracy_score(labels_array, p)
        precision = precision_score(labels_array, p)
        recall = recall_score(labels_array, p)
        f1 = f1_score(labels_array, p)
        mcc = matthews_corrcoef(labels_array, p)

        print(f'Valid ACC: {acc}, F-1: {f1}, AUC: {test_auc}, Precision: {precision}, Recall: {recall}', 'MCC:', mcc)
        
        return acc, f1, test_auc, precision, recall, mcc, val_loss

    
if __name__ == '__main__':
    my_lib = Ecoli()
    my_lib.train()

