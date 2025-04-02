import re
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datasets.MMSol_Dataset import collate_fn, MMSol_Dataset

def read_fasta_to_dataframe(file_path1):
    '''
    Parsing fasta files
    '''
    records = SeqIO.to_dict(SeqIO.parse(file_path1, "fasta"))

    pattern = re.compile(r'label=(\d+) label_noise=(\d+) feature=\[([^\]]+)\] GO=\[([^\]]+)\]')

    data_source = []
    sequences = []
    label_tag = []
    label_noise = []
    feature = []
    GO = []
    
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
    
    df = pd.DataFrame({
        'data_source': data_source,
        'sequence': sequences,
        'label': label_tag,
        'label_noise': label_noise, 
        'feature': feature,
        'GO': GO
    })
    
    return df

def range_get_reg(model_path, fasta_path, max_len, edge_path, node_path, GO_path, output_path):   
    '''
    Delimit the range of noise samples in the training set by pretrained regression model

    The sample which may have noise is marked as 2, and the sample which may have no noise is marked as 1
    '''      
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    model = torch.load(model_path, map_location="cuda:0")  
    model = model.to(device)

    model.eval()
    
    test_dataset = MMSol_Dataset(fasta_path, max_pad_len=max_len, 
                                 edge_fea_path=edge_path, 
                                 node_fea_path=node_path, 
                                 GO_fea_path=GO_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    predicted_list = []
    labels_list = []
    sample_ids = []
    label_noise_dict = {}

    loop = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(loop, 0):
            
            id, inputs, attention_ids, feature, GO_fea, labels_cpu, label_noise, sequences_feature, sequences_mask, graph_feature, graph_mask, softlabels, index = data

            inputs = inputs.to(device)
            attention_ids = attention_ids.to(device)
            feature = feature.to(device)
            GO_fea = GO_fea.to(device)
            labels = labels_cpu.to(device).float()  
            sequences_feature = sequences_feature.to(device)
            sequences_mask = sequences_mask.to(device)
            graph_feature = graph_feature.to(device)
            graph_mask = graph_mask.to(device)

            outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask, graph_mask)
            
            predicted = (outputs >= 0.5).cpu().numpy().flatten()  
            labels_cpu = labels.cpu().numpy().flatten() 

            sample_ids += list(id)
            predicted_list += predicted.tolist()
            labels_list += labels_cpu.tolist()

            for j, true_label in enumerate(labels_cpu):
                if true_label == 1:
                    label_noise_dict[id[j]] = label_noise[j]  
                else:
                    if predicted[j] == 0:
                        label_noise_dict[id[j]] = 1  
                    else:
                        label_noise_dict[id[j]] = 2  
    e
    df = read_fasta_to_dataframe(fasta_path)

    for idx, row in df.iterrows():
        if row['data_source'] in label_noise_dict:
            df.at[idx, 'label_noise'] = label_noise_dict[row['data_source']]

    updated_records = []

    for _, row in df.iterrows():
        updated_description = f"label={row['label']} label_noise={row['label_noise']} feature={row['feature']} GO={row['GO']}"
        
        updated_record = SeqRecord(Seq(str(row['sequence'])), id=row['data_source'], description=updated_description)
        updated_records.append(updated_record)

    SeqIO.write(updated_records, output_path, "fasta")
    print(f"Updated FASTA file saved to {output_path}")

    return predicted_list, labels_list, label_noise_dict

def range_get_cls(model_path, fasta_path, max_len, edge_path, node_path, GO_path, output_path, MMSol_Dataset, collate_fn, read_fasta_to_dataframe):
    '''
    Delimit the range of noise samples in the training set by pretrained classfication model
    
    The sample which may have noise is marked as 2, and the sample which may have no noise is marked as 1
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device).to(device)
    model.eval()
    
    test_dataset = MMSol_Dataset(fasta_path, max_pad_len=max_len, 
                                 edge_fea_path=edge_path, 
                                 node_fea_path=node_path, 
                                 GO_fea_path=GO_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    label_noise_dict = {}
    predicted_list = []
    labels_list = []
    sample_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            id, inputs, attention_ids, feature, GO_fea, labels_cpu, label_noise, sequences_feature, sequences_mask, graph_feature, graph_mask, softlabels, index = batch

            inputs = inputs.to(device)
            attention_ids = attention_ids.to(device)
            feature = feature.to(device)
            GO_fea = GO_fea.to(device)
            labels = labels_cpu.to(device).float()
            sequences_feature = sequences_feature.to(device)
            sequences_mask = sequences_mask.to(device)
            graph_feature = graph_feature.to(device)
            graph_mask = graph_mask.to(device)

            outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask, graph_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
            predicted = (probs >= 0.5).astype(int)
            labels_cpu = labels.cpu().numpy().flatten()

            sample_ids += list(id)
            predicted_list += predicted.tolist()
            labels_list += labels_cpu.tolist()

            for j, true_label in enumerate(labels_cpu):
                if true_label == 1:
                    label_noise_dict[id[j]] = label_noise[j]  # 保留原标签
                else:
                    label_noise_dict[id[j]] = 1 if predicted[j] == 0 else 2  # 预测为 0 高置信，为 1 可能是错的

    df = read_fasta_to_dataframe(fasta_path)
    for idx, row in df.iterrows():
        if row['data_source'] in label_noise_dict:
            df.at[idx, 'label_noise'] = label_noise_dict[row['data_source']]

    updated_records = []
    for _, row in df.iterrows():
        feature_str = ",".join(map(str, row['feature']))
        GO_str = ",".join(map(str, row['GO']))
        updated_description = f"label={row['label']} label_noise={row['label_noise']} feature=[{feature_str}] GO=[{GO_str}]"
        updated_record = SeqRecord(Seq(row['sequence']), id=row['data_source'], description=updated_description)
        updated_records.append(updated_record)

    SeqIO.write(updated_records, output_path, "fasta")
    print(f"Updated FASTA written to: {output_path}")

    return predicted_list, labels_list, label_noise_dict

if __name__ == '__main__':
    model_path = None  # Guidance Model trained on the eSOL dataset
    fasta_path = './data/noise/train.fasta'
    max_len = 200
    edge_path = './data/noise/noise_graph/train_LPE_5_1.pkl'
    node_path = './data/noise/noise_graph/train_node.pkl'
    GO_path = './data/noise/noise_go/train_go.pkl'
    output_path = './data/noise/train_range.fasta'

    range_get_reg(model_path, fasta_path, max_len, edge_path, node_path, GO_path, output_path)

