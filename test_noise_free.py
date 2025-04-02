import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve

from datasets.MMSol_Dataset_noise_free import MMSol_Dataset, collate_fn
from utils.SparseGO.utils_conform import *
from models.MMSol import Model
from torch.utils.data import DataLoader
from configs import config_noise_free_cls

from tqdm import tqdm
import argparse

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


parser = argparse.ArgumentParser(description='Test for MMSol')

parser.add_argument('--batch_size', default=config_noise_free_cls.batch_size, type=int, help='Batch size')
parser.add_argument('--seed', default=config_noise_free_cls.seed, type=int, help='Random seed')
parser.add_argument('--num_workers', default=config_noise_free_cls.num_workers, type=int, help='Number of workers for data loading')
parser.add_argument('--gpu', default=config_noise_free_cls.gpu, type=int, help='GPU number')
parser.add_argument('--test_dataset_path', default=config_noise_free_cls.test_dataset_path, type=str, help='Path for test dataset')
parser.add_argument('--max_pad_len', default=config_noise_free_cls.max_pad_len, type=int, help='Max pad length for sequence')
parser.add_argument('--model_path', default=config_noise_free_cls.model_path, type=str, help='Path for model')
parser.add_argument('--protein2id', default=config_noise_free_cls.protein2id, type=str, help='protein2id')
parser.add_argument('--protein2ont', default=config_noise_free_cls.protein2ont, type=str, help='protein2ont')

args = parser.parse_args()


class Ecoli(MMSol_Dataset):

    def __init__(self):
        global args
        self.batch_size = args.batch_size
        self.test_dataset = args.test_dataset_path
        self.model_path = args.model_path
        self.seed = args.seed
        self.num_workers = args.num_workers  
        self.max_pad_len = args.max_pad_len
        self.gpu = args.gpu
        self.protein2id = args.protein2id
        self.protein2ont = args.protein2ont
    def test(self, model=None):         
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")

        # # model = Model()
        model = torch.load(self.model_path, map_location=torch.device('cuda:0')) 
        model = model.to(device)
        model.eval()

        # # -----SparseGO-----
        # protein2id_mapping = load_mapping(self.protein2id)
        # dG, terms_pairs, proteins_terms_pairs = load_ontology(self.protein2ont, protein2id_mapping)
        # sorted_pairs, level_list, level_number = sort_pairs(
        #     proteins_terms_pairs, terms_pairs, dG, protein2id_mapping)
        # layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)  

        # # -----Model Define----- 
        # model = Model(layer_connections=layer_connections)
        # model.load_state_dict(torch.load(self.model_path))  
        model = model.to(device)
        
        # -----Dataset-----
        test_dataset = MMSol_Dataset(self.test_dataset, max_pad_len=self.max_pad_len, 
                                     edge_fea_path='./data/noise_free/noise_free_graph/test_LPE_5_1.pkl', 
                                     node_fea_path='./data/noise_free/noise_free_graph/test_node.pkl', 
                                     GO_fea_path='./data/noise_free/noise_free_go/test_go.pkl')
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

        print(f'Test ACC: {acc}, F-1: {f1}, AUC: {test_auc}, Precision: {precision}, Recall: {recall}', 'MCC:', mcc)
        return acc, f1, test_auc, precision, recall, mcc

    
if __name__ == '__main__':
    exp = Ecoli()
    exp.test()
