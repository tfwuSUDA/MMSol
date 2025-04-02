import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

from datasets.MMSol_Dataset_noise_free_reg import MMSol_Dataset, collate_fn
from utils.SparseGO.utils_conform import *
from models.MMSol_reg import Model
from models.MMSol_reg import *
from torch.utils.data import DataLoader, Subset
from configs import config_noise_free_reg

from tqdm import tqdm
import argparse
import os
import torch.nn.parallel
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser(description='Training for MMSol')

parser.add_argument('--epochs', default=config_noise_free_reg.epochs, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=config_noise_free_reg.batch_size, type=int, help='Batch size')
parser.add_argument('--lr', default=config_noise_free_reg.lr, type=float, help='Learning rate')
parser.add_argument('--seed', default=config_noise_free_reg.seed, type=int, help='Random seed')
parser.add_argument('--num_workers', default=config_noise_free_reg.num_workers, type=int, help='Number of workers for data loading')
parser.add_argument('--weight_decay', default=config_noise_free_reg.weight_decay, type=float, help='Weight decay for optimizer')
parser.add_argument('--gpu', default=config_noise_free_reg.gpu, type=int, help='GPU number')
parser.add_argument('--train_dataset_path', default=config_noise_free_reg.train_dataset_path, type=str, help='Path for train dataset')
parser.add_argument('--max_pad_len', default=config_noise_free_reg.max_pad_len, type=int, help='Max pad length for sequence')
parser.add_argument('--model_path', default=config_noise_free_reg.model_path, type=str, help='Path for model')
parser.add_argument('--save_path', default=config_noise_free_reg.save_path, type=str, help='Path for save the best model')
parser.add_argument('--protein2id', default=config_noise_free_reg.protein2id, type=str, help='protein2id')
parser.add_argument('--protein2ont', default=config_noise_free_reg.protein2ont, type=str, help='protein2ont')

args = parser.parse_args()


class Ecoli(MMSol_Dataset):

    def __init__(self):
        global args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.train_dataset  = args.train_dataset_path
        self.model_path = args.model_path
        self.seed = args.seed
        self.num_workers = args.num_workers  
        self.save_path = args.save_path
        self.weight_decay = args.weight_decay
        self.max_pad_len = args.max_pad_len
        self.gpu = args.gpu
        self.protein2id = args.protein2id
        self.protein2ont = args.protein2ont


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
        
        # -----Dataset-----
        train_dataset = MMSol_Dataset(self.train_dataset, max_pad_len=self.max_pad_len, 
                                      edge_fea_path='./data/noise_free/noise_free_graph/train_LPE_5_1.pkl', 
                                      node_fea_path='./data/noise_free/noise_free_graph/train_node.pkl', 
                                      GO_fea_path='./data/noise_free/noise_free_go/train_go.pkl')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.num_workers, collate_fn=collate_fn)
                
        # -----Loss-----
        criterion = nn.MSELoss().to(device)  
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)  

        # -----train-----
        for epoch in range(1,self.epochs+1): 
            model.train()

            train_loss = 0

            loop = tqdm(total=len(train_loader), leave=False)

            all_preds = []
            all_labels = []

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

                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs_cpu = outputs.detach().cpu().numpy()
                labels_cpu_np = labels_cpu.numpy()
                
                all_preds.extend(outputs_cpu)
                all_labels.extend(labels_cpu_np)
                
                train_loss += loss.item()
                    
                r2 = r2_score(labels_cpu_np, outputs_cpu)
                rmse = np.sqrt(mean_squared_error(labels_cpu_np, outputs_cpu))

                loop.set_description(f"Epoch {epoch} | Loss {loss.item():.5f}")
                loop.set_postfix(loss=loss.item(), r2=r2, rmse=rmse)
                if np.isnan(loss.item()):
                    print('nan')
                    break
                loop.update()  

            train_r2 = r2_score(all_labels, all_preds)
            train_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            print(f"Train R2: {train_r2:.5f}")
            print(f"Train RMSE: {train_rmse:.5f}")
                
        model_path = os.path.join(self.save_path, f"{epoch}_eSOL_reg.pth")
        torch.save(model, model_path)
    
    def train_cv(self):
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

        # -----Dataset-----
        train_dataset = MMSol_Dataset(self.train_dataset, max_pad_len=self.max_pad_len, 
                                      edge_fea_path='./data/noise_free/noise_free_graph/train_LPE_5_1.pkl', 
                                      node_fea_path='./data/noise_free/noise_free_graph/train_node.pkl', 
                                      GO_fea_path='./data/noise_free/noise_free_go/train_go.pkl')

        # -----Cross-validation Setup-----
        kfold = KFold(n_splits=5, shuffle=True, random_state=2333)

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
            
            # -----Loss-----
            criterion = nn.MSELoss().to(device)  
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-3)  


            train_subdataset = Subset(train_dataset, train_idx)
            val_subdataset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subdataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_subdataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
                
            model.train()

            # Best models for validation AUC and validation loss
            best_loss = np.inf
            best_loss_model = None
            best_rmse = np.inf

            train_r2_ls = []
            train_rmse_ls = []
            best_rmse_model = None

            # -----Train for current fold-----
            for epoch in range(1, self.epochs + 1):
                model.train()
                train_loss = 0

                loop = tqdm(total=len(train_loader), leave=False)

                all_preds = []
                all_labels = []

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
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    outputs_cpu = outputs.detach().cpu().numpy()
                    labels_cpu_np = labels_cpu.numpy()
                    
                    all_preds.extend(outputs_cpu)
                    all_labels.extend(labels_cpu_np)

                    train_loss += loss.item()
                    
                    r2 = r2_score(labels_cpu_np, outputs_cpu)
                    rmse = np.sqrt(mean_squared_error(labels_cpu_np, outputs_cpu))

                    loop.set_description(f"Epoch {epoch} | Loss {loss.item():.5f}")
                    loop.set_postfix(loss=loss.item(), r2=r2, rmse=rmse)
                    if np.isnan(loss.item()):
                        print('nan')
                        break
                    loop.update() 

                train_r2 = r2_score(all_labels, all_preds)
                train_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
                print(f"Train R2: {train_r2:.5f}")
                print(f"Train RMSE: {train_rmse:.5f}")
                train_r2_ls.append(train_r2)
                train_rmse_ls.append(train_rmse)

                # -----Validation for current fold-----
                model.eval()  # switch to evaluation mode
                val_loss = 0
                predicted_list = []
                labels_list = []
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
                        loss = criterion(outputs, labels)
                        sample_ids += list(id)
                        predicted_list += outputs.cpu().numpy().tolist()
                        labels_list += labels.cpu().numpy().tolist()

                        val_loss += loss.item()

                # Calculate metrics
                predicted_array = np.array(predicted_list)
                labels_array = np.array(labels_list)

                val_r2 = r2_score(labels_array, predicted_array)
                val_rmse = np.sqrt(mean_squared_error(labels_array, predicted_array))

                print(f'Valid R2: {val_r2:.5f}, RMSE: {val_rmse:.5f}')

                metrics_path = os.path.join(self.save_path, f"loss_model_fold_{fold+1}.txt")
                with open(metrics_path, 'a') as f:
                    f.write(f"Epoch: {epoch} Valid R2: {val_r2:.5f} Valid RMSE: {val_rmse:.5f} Valid Loss: {val_loss} Train Loss: {train_loss}\n")

                # Save the model 
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_rmse_model = model.state_dict()

                    model_path = os.path.join(self.save_path, f"best_rmse_model_fold_{fold+1}.pth")
                    torch.save(best_rmse_model, model_path)
                    print(f"Best RMSE model for fold {fold+1} saved.")

                    metrics_path = os.path.join(self.save_path, f"best_rmse_model_fold_{fold+1}.txt")
                    with open(metrics_path, 'w') as f:
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Valid R2: {val_r2:.5f}\n")
                        f.write(f"Valid RMSE: {val_rmse:.5f}\n")

                if val_loss / len(val_loader) < best_loss:
                    best_loss = val_loss / len(val_loader)
                    best_loss_model = model.state_dict()

                    model_path = os.path.join(self.save_path, f"best_loss_model_fold_{fold+1}.pth")
                    torch.save(best_loss_model, model_path)
                    print(f"Best Loss model for fold {fold+1} saved.")

                    metrics_path = os.path.join(self.save_path, f"best_loss_model_fold_{fold+1}.txt")
                    with open(metrics_path, 'w') as f:
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Valid R2: {val_r2:.5f}\n")
                        f.write(f"Valid RMSE: {val_rmse:.5f}\n")
    
if __name__ == '__main__':
    my_lib = Ecoli()
    # my_lib.train_cv()
    my_lib.train_total()

