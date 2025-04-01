import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve, auc

from lib.MMSol_Dataset_noise_free import MMSol_Dataset, collate_fn
from SparseGO.utils_conform import *
from models.MMSol import Model
from torch.utils.data import DataLoader, Subset
from config import config_noise_free_cls

from tqdm import tqdm
import argparse
import os
import torch.nn.parallel
import random
from Bio import SeqIO
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser(description='Training for MMSol')

parser.add_argument('--epochs', default=config_noise_free_cls.epochs, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=config_noise_free_cls.batch_size, type=int, help='Batch size')
parser.add_argument('--lr', default=config_noise_free_cls.lr, type=float, help='Learning rate for noise data in part 1')
parser.add_argument('--seed', default=config_noise_free_cls.seed, type=int, help='Random seed')
parser.add_argument('--num_workers', default=config_noise_free_cls.num_workers, type=int, help='Number of workers for data loading')
parser.add_argument('--weight_decay', default=config_noise_free_cls.weight_decay, type=float, help='Weight decay for optimizer')
parser.add_argument('--gpu', default=config_noise_free_cls.gpu, type=int, help='GPU number')
parser.add_argument('--train_dataset_path', default=config_noise_free_cls.train_dataset_path, type=str, help='Path for train dataset')
parser.add_argument('--test_dataset_path', default=config_noise_free_cls.test_dataset_path, type=str, help='Path for test dataset')
parser.add_argument('--max_pad_len', default=config_noise_free_cls.max_pad_len, type=int, help='Max pad length for sequence')
parser.add_argument('--model_path', default=config_noise_free_cls.model_path, type=str, help='Path for model')
parser.add_argument('--save_path', default=config_noise_free_cls.save_path, type=str, help='Path for save the best model')
parser.add_argument('--protein2id', default=config_noise_free_cls.protein2id, type=str, help='protein2id')
parser.add_argument('--protein2ont', default=config_noise_free_cls.protein2ont, type=str, help='protein2ont')

args = parser.parse_args()


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def read_fasta_to_dataframe(file_path1):
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


class Ecoli(MMSol_Dataset):

    def __init__(self):
        global args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.train_dataset  = args.train_dataset_path
        self.test_dataset = args.test_dataset_path
        self.model_path = args.model_path
        self.seed = args.seed
        self.save_path = args.save_path
        self.num_workers = args.num_workers  
        self.save_path = args.save_path
        self.weight_decay = args.weight_decay
        self.max_pad_len = args.max_pad_len
        self.gpu = args.gpu
        self.protein2id = args.protein2id
        self.protein2ont = args.protein2ont

        self.y_file = self.save_path + "y.npy"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.save_path+'record'):
            os.makedirs(self.save_path+'record')


    def train_total(self):
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

        # -----SparceGO-----
        
        protein2id_mapping = load_mapping(self.protein2id)
        dG, terms_pairs, proteins_terms_pairs = load_ontology(self.protein2ont, protein2id_mapping)
        sorted_pairs, level_list, level_number = sort_pairs(
            proteins_terms_pairs, terms_pairs, dG, protein2id_mapping)
        layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)  # 添加虚拟节点

        # -----Model Define----- 
        model = Model(layer_connections=layer_connections)
        # print(model)

        model = model.to(device)
        model.train()

        # -----Dataset-----
                
        train_dataset = MMSol_Dataset(self.train_dataset, max_pad_len=self.max_pad_len, 
                                      edge_fea_path='./data/noise_free/eSOL_edge/train_LPE_5_1.pkl', 
                                      node_fea_path='./data/noise_free/eSOL_edge/train_node.pkl', 
                                      GO_fea_path='./data/noise_free/eSOL_go/train_go.pkl')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.num_workers, collate_fn=collate_fn)
                
        # -----Loss-----
        total_labels = train_dataset.get_data_labels()  
        total_samples = len(total_labels)
        num_pos_samples = (total_labels == 1).sum().item()  
        num_neg_samples = (total_labels == 0).sum().item()
        
        pos_weight = total_samples / (2 * num_pos_samples)  
        neg_weight = total_samples / (2 * num_neg_samples)  
        print(f'pos_weight: {pos_weight}, neg_weight: {neg_weight}')
        weights = torch.tensor([neg_weight, pos_weight]).to(device)
        criterion = nn.NLLLoss(weight=weights).to(device) 
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)  

        train_acc_ls = []

        # -----train-----
        for epoch in range(1,self.epochs+1): 
            model.train()

            train_loss = 0
            corrects_epoch = 0  
            total_epoch = 0  

            loop = tqdm(total=len(train_loader), leave=False)

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
                outputs= model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask,  graph_mask)
                labels = labels.long()
                log_outputs = torch.log_softmax(outputs, 1).float()

                loss = criterion(log_outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
            train_acc_ls.append(train_acc)
            
                
        model_path = os.path.join(self.save_path, f"{epoch}_eSOL.pth")
        torch.save(model, model_path)
    
    def train_cv(self):
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

        # -----Dataset-----
        train_dataset = MMSol_Dataset(self.train_dataset, max_pad_len=self.max_pad_len, 
                                      edge_fea_path='./data/noise_free/eSOL_edge/train_LPE_5_1.pkl', 
                                      node_fea_path='./data/noise_free/eSOL_edge/train_node.pkl', 
                                      GO_fea_path='./data/noise_free/eSOL_go/train_go.pkl')

        # -----Cross-validation Setup-----
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"Training fold {fold+1}/{5}")
            # -----SparseGO-----
            protein2id_mapping = load_mapping(self.protein2id)
            dG, terms_pairs, proteins_terms_pairs = load_ontology(self.protein2ont, protein2id_mapping)
            sorted_pairs, level_list, level_number = sort_pairs(
                proteins_terms_pairs, terms_pairs, dG, protein2id_mapping)
            layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)  

            # -----Model Define----- 
            model = Model(layer_connections=layer_connections)
            model = model.to(device)
            model.train()

            
            # -----Dataset-----
            train_subdataset = Subset(train_dataset, train_idx)
            val_subdataset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subdataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_subdataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
            
            # -----Loss-----
            criterion = nn.NLLLoss().to(device) 
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)
            
            model.train()

            # Best models for validation AUC and validation loss
            best_auc = -np.inf
            best_loss = np.inf
            best_auc_model = None
            best_loss_model = None

            # -----Train for current fold-----
            for epoch in range(1, self.epochs + 1):
                model.train()
                train_loss = 0
                corrects_epoch = 0
                total_epoch = 0

                loop = tqdm(total=len(train_loader), leave=False)

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

                    outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask, graph_mask)
                    labels = labels.long()
                    log_outputs = torch.log_softmax(outputs, 1).float()

                    loss = criterion(log_outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    preds_noise = torch.argmax(outputs, dim=1)
                    corrects = (preds_noise == labels).float().sum()
                    corrects_epoch += corrects.item()
                    total_epoch += inputs.size(0)
                    train_loss += loss.item()

                    loop.set_description(f"Epoch {epoch} | Loss {loss.item():.5f}")
                    loop.set_postfix(loss=loss.item(), acc=corrects.item()/inputs.size(0))   
                    loop.update()

                train_acc = corrects_epoch / total_epoch
                print(f"Train Accuracy for fold {fold+1}, Epoch {epoch}: {train_acc:.5f}")

                # -----Validation for current fold-----
                model.eval()  
                val_loss = 0
                predicted_list = []
                labels_list = []
                prob_all = []
                sample_ids = []

                with torch.no_grad():
                    for batch in val_loader:
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

                        outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask, graph_mask)
                        labels = labels.long()
                        log_outputs = torch.log_softmax(outputs, 1).float()

                        loss = criterion(log_outputs, labels)
                        val_loss += loss.item()

                        outputs = torch.softmax(outputs, 1)
                        prob_all.extend(outputs[:,1].cpu().numpy()) 
                        predicted = torch.argmax(outputs, dim=1)

                        sample_ids += list(id)
                        predicted_list += predicted.tolist()

                        labels_list += labels.tolist()

                # Calculate AUC
                predicted_array = np.array(predicted_list)
                labels_array = np.array(labels_list)
                prob_all = np.array(prob_all)

                val_acc = accuracy_score(labels_array, predicted_array)
                fpr, tpr, thresholds = roc_curve(labels_array, prob_all)
                roc_auc = auc(fpr, tpr)

                print(f"Validation Accuracy for fold {fold+1}, Epoch {epoch}: {val_acc:.5f}")
                print(f"Validation Loss for fold {fold+1}, Epoch {epoch}: {val_loss / len(val_loader):.5f}")
                print(f"Validation AUC for fold {fold+1}, Epoch {epoch}: {roc_auc:.5f}")

                metrics_path = os.path.join(self.save_path, f"loss_model_fold_{fold+1}.txt")
                with open(metrics_path, 'a') as f:
                    f.write(f"Epoch: {epoch} Valid ACC: {val_acc:.5f} Valid AUC: {roc_auc:.5f} Valid Loss: {val_loss} Train Loss: {train_loss}\n")

                # Save the model if AUC is improved or if Loss is improved
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_auc_model = model.state_dict()
                    model_path = os.path.join(self.save_path, f"best_rmse_model_fold_{fold+1}.pth")
                    torch.save(best_auc_model, model_path)
                    print(f"Best AUC model for fold {fold+1} saved.")

                    metrics_path = os.path.join(self.save_path, f"best_auc_model_fold_{fold+1}.txt")
                    with open(metrics_path, 'w') as f:
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Valid AUC: {roc_auc:.5f}\n")

                if val_loss / len(val_loader) < best_loss:
                    best_loss = val_loss / len(val_loader)
                    best_loss_model = model.state_dict()

                    model_path = os.path.join(self.save_path, f"best_loss_model_fold_{fold+1}.pth")
                    torch.save(best_loss_model, model_path)
                    print(f"Best Loss model for fold {fold+1} saved.")

                    metrics_path = os.path.join(self.save_path, f"best_loss_model_fold_{fold+1}.txt")
                    with open(metrics_path, 'w') as f:
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Valid LOSS: {best_loss:.5f}\n")

    
    def test(self, model=None):         
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

        # # model = Model()
        # model = torch.load(self.model_path) 
        # model = model.to(device)
        # model.eval()

        # -----SparseGO-----
        protein2id_mapping = load_mapping(self.protein2id)
        dG, terms_pairs, proteins_terms_pairs = load_ontology(self.protein2ont, protein2id_mapping)
        sorted_pairs, level_list, level_number = sort_pairs(
            proteins_terms_pairs, terms_pairs, dG, protein2id_mapping)
        layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)  

        # -----Model Define----- 
        model = Model(layer_connections=layer_connections)
        model.load_state_dict(torch.load(self.model_path))  
        model = model.to(device)
        
        # -----Dataset-----
        test_dataset = MMSol_Dataset(self.test_dataset, max_pad_len=self.max_pad_len, 
                                     edge_fea_path='./data/noise_free/eSOL_edge/test_LPE_5_1.pkl',
                                     node_fea_path='./data/noise_free/eSOL_edge/test_node.pkl', 
                                     GO_fea_path='./data/noise_free/eSOL_go/test_go.pkl')
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.num_workers, collate_fn=collate_fn)

        predicted_list = []
        labels_list = []
        prob_all = []
        sample_ids = []

        loop = tqdm(test_loader)
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

                outputs= model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask,  graph_mask)

                outputs = torch.softmax(outputs, 1)
                prob_all.extend(outputs[:,1].cpu().numpy()) 
                predicted = torch.argmax(outputs, dim=1)

                sample_ids += list(id)
                predicted_list += predicted.tolist()

                labels_list += labels.tolist()
                
        predicted_array = np.array(predicted_list)
        labels_array = np.array(labels_list)
        prob_all = np.array(prob_all)

        test_auc = roc_auc_score(labels_array, prob_all)
        fpr, tpr, thresholds = roc_curve(labels_array, prob_all)
        roc_auc = auc(fpr, tpr)

        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_idx]
        print(f'Best Threshold: {best_threshold}')

        p = prob_all.copy()
        p[p>=best_threshold] = 1
        p[p<best_threshold] = 0

        acc = accuracy_score(labels_array, p)
        precision = precision_score(labels_array, p)
        recall = recall_score(labels_array, p)
        f1 = f1_score(labels_array, p)
        mcc = matthews_corrcoef(labels_array, p)

        auc_raw = roc_auc_score(labels_array, prob_all)
        precision_raw = precision_score(labels_array, predicted_array)
        recall_raw = recall_score(labels_array, predicted_array)
        f1_raw = f1_score(labels_array, predicted_array)
        acc_raw = accuracy_score(labels_array, predicted_array)
        mcc_raw = matthews_corrcoef(labels_array, predicted_array)

        print(f'Test ACC raw: {acc_raw}, F-1 raw: {f1_raw}, AUC raw: {auc_raw}, Precision raw: {precision_raw}, Recall raw: {recall_raw}', 'MCC raw:', mcc_raw)
        print(f'Test ACC: {acc}, F-1: {f1}, AUC: {test_auc}, Precision: {precision}, Recall: {recall}', 'MCC:', mcc)
        return acc, f1, test_auc, precision, recall, mcc

    
if __name__ == '__main__':
    my_lib = Ecoli()
    my_lib.train_cv()
    # my_lib.train_total()
    # my_lib.test()
