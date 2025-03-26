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
from lib.MMSol_Dataset import collate_fn, MMSol_Dataset

def read_fasta_to_dataframe(file_path1):
    # 读取 FASTA 文件并解析
    records = SeqIO.to_dict(SeqIO.parse(file_path1, "fasta"))

    # 正则表达式用于提取 label 和 label_noise 等信息
    pattern = re.compile(r'label=(\d+) label_noise=(\d+) feature=\[([^\]]+)\] GO=\[([^\]]+)\]')

    # 初始化保存的列表
    data_source = []
    sequences = []
    label_tag = []
    label_noise = []
    feature = []
    GO = []
    
    # 遍历每个 FASTA 记录
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
    
    # 创建并返回 DataFrame
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
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    # 加载回归模型
    model = torch.load(model_path, map_location="cuda:0")  
    model = model.to(device)

    model.eval()
    
    # 读取数据集
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

            # 将每个数据移动到适当的设备
            inputs = inputs.to(device)
            attention_ids = attention_ids.to(device)
            feature = feature.to(device)
            GO_fea = GO_fea.to(device)
            labels = labels_cpu.to(device).float()  # 转换为浮点型
            sequences_feature = sequences_feature.to(device)
            sequences_mask = sequences_mask.to(device)
            graph_feature = graph_feature.to(device)
            graph_mask = graph_mask.to(device)

            # 获取模型预测的输出
            outputs = model(inputs, attention_ids, feature, GO_fea, sequences_feature, graph_feature, sequences_mask, graph_mask)
            
            # 获取预测结果
            predicted = (outputs >= 0.5).cpu().numpy().flatten()  # 用 0.5 阈值将输出转换为 0 或 1
            labels_cpu = labels.cpu().numpy().flatten()  # 获取真实标签

            # 将 ID、预测结果和真实标签添加到对应的列表中
            sample_ids += list(id)
            predicted_list += predicted.tolist()
            labels_list += labels_cpu.tolist()

            # 根据真实标签更新 label_noise_dict
            for j, true_label in enumerate(labels_cpu):
                if true_label == 1:
                    # 如果真实标签是 1，跳过
                    label_noise_dict[id[j]] = label_noise[j]  # 保持原始 label_noise
                else:
                    # 如果真实标签是 0，根据预测结果更新 label_noise
                    if predicted[j] == 0:
                        label_noise_dict[id[j]] = 1  # 如果预测结果是 0，label_noise 设置为 1
                    else:
                        label_noise_dict[id[j]] = 2  # 如果预测结果是 1，label_noise 设置为 2
    
    # 读取 fasta 文件并转换成 DataFrame
    df = read_fasta_to_dataframe(fasta_path)

    # 根据 label_noise_dict 更新 df 中的 label_noise 列
    for idx, row in df.iterrows():
        # 查找 id 对应的 label_noise 值并更新
        if row['data_source'] in label_noise_dict:
            df.at[idx, 'label_noise'] = label_noise_dict[row['data_source']]

    # 保存更新后的 DataFrame 为原始格式的 FASTA 文件
    updated_records = []

    # 使用 df 生成 FASTA 文件的记录
    for _, row in df.iterrows():
        # 构建新的 description，其中包含 label 和更新后的 label_noise
        updated_description = f"label={row['label']} label_noise={row['label_noise']} feature={row['feature']} GO={row['GO']}"
        
        # 创建新的 SeqRecord
        updated_record = SeqRecord(Seq(str(row['sequence'])), id=row['data_source'], description=updated_description)
        updated_records.append(updated_record)

    # 保存为新的 FASTA 文件
    SeqIO.write(updated_records, output_path, "fasta")
    print(f"Updated FASTA file saved to {output_path}")

    return predicted_list, labels_list, label_noise_dict

    # # 将结果保存为 CSV 文件
    # results_df = pd.DataFrame({
    #     'ID':123456
    #  sample_ids,
    #     'Label': labels_list,
    #     'Pred': predicted_list,
    #     'Label_noise': label_noise_list
    # })
    # results_df.to_csv('/linhaitao/xujia/MMSol/Best_ckpt/noise_free_reg/reg_21_55_range.csv', index=False)  # 保存到CSV文件
    # print("Results saved to 'reg_21_55_range.csv'")

    return predicted_list, labels_list, label_noise_list

if __name__ == '__main__':
    # model_path = '/linhaitao/xujia/MMSol/Best_ckpt/noise_free_reg/reg_21.3_55.5.pth'
    model_path = '/linhaitao/xujia/MMSol/lib_output/range/125_eSOL_range.pth'
    fasta_path = './data/noise_new/noise_train.fasta'
    max_len = 200
    edge_path = './data/noise/noise_edge/nesg_train_LPE_head_5_1.pkl'
    node_path = './data/noise/noise_edge/nesg_train_node_features_dssp.pkl'
    GO_path = './data/noise/noise_go/nesg_train_uniprot_go_conform_onehot_concat.pkl'
    output_path = './data/noise_new/noise_train_range_125.fasta'

    range_get_reg(model_path, fasta_path, max_len, edge_path, node_path, GO_path, output_path)

