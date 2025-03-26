import os
import pickle
import re
import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix
import torch
from Bio import SeqIO
import subprocess
from sklearn.discriminant_analysis import StandardScaler

# —————————————————— Adjacency Matrix ——————————————————

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
    # print("PDB: "+pdbdir+"\tdistance matrix shape: "+str(np.matrix(dismatrix).shape))
    dismatrix = np.where(np.array(dismatrix), 1, 0)
    contactmatrix = coo_matrix(dismatrix)
    edge_index = torch.LongTensor(np.vstack((contactmatrix.row,contactmatrix.col)))
    return dismatrix, edge_index

# —————————————————— LPE ——————————————————

def laplacian_matrix(adj_matrix):
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix
    
    return laplacian

def process_adj_matrices(pkl_file, output_file):
    with open(pkl_file, 'rb') as f:
        adj_matrices_dict = pickle.load(f)
    laplacian_dict = {}

    for id, adj in adj_matrices_dict.items():
        print(f'Processing sample with id: {id}')
        adj_np = adj.astype(np.int32)
        laplacian = laplacian_matrix(adj_np)
        laplacian_dict[id] = laplacian.astype(np.int16)

    with open(output_file, 'wb') as f:
        pickle.dump(laplacian_dict, f)

# —————————————————— Node fea ——————————————————

def load_features(sequence, protein_id, data=None):
    if data is None:
        print("Error: data is None")
        return
    blosum_dict = load_blosum()
    aaphy7_dict = load_aaphy7()
    blosum = np.array([blosum_dict[amino] for amino in sequence])
    aaphy7 = np.array([aaphy7_dict[amino] for amino in sequence])
    
    protein_features = data.get(protein_id)
    if protein_features is None:
        print(f'Protein {protein_id} not found in the dataset.')
        return None

    features_list = []
    acc_values = [features['acc'] for residue_num, features in protein_features.items()]

    if not acc_values:
        print(f'No ACC values found for protein {protein_id}.')
        return None
    scaler = StandardScaler()
    acc_values = scaler.fit_transform(np.array(acc_values).reshape(-1, 1)).flatten()

    for (residue_num, features), acc in zip(protein_features.items(), acc_values):
        tco = features['tco']
        kappa = features['kappa'] / 360
        alpha = features['alpha'] / 360
        phi = features['phi'] / 360
        psi = features['psi'] / 360
        features_list.append([acc, tco, kappa, alpha, phi, psi])

    feature_matrix = np.array(features_list)
    feature_matrix = np.concatenate([blosum, aaphy7, feature_matrix], axis=1)
    
    return feature_matrix

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

def node_fea_process(fasta_file, output_file, data=None):
    feature_matrices = {}
    count = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        sequence = str(record.seq)

        feature_matrix = load_features(sequence, protein_id, data)
        if feature_matrix is None:
            count += 1
        feature_matrices[protein_id] = feature_matrix
    print(f'Error count: {count}')
    with open(output_file, 'wb') as file:
        pickle.dump(feature_matrices, file)

# —————————————————— H-bond ——————————————————

def update_adjacency_matrix(adjacency_matrix, protein_features, w1, w2, protein_id):
    for residue_num, features in protein_features.items():
        nh_o_1  = features['nh_o_1']

        try:
            for index, energy in [nh_o_1]:
                if index != 0:
                    
                    adjacency_matrix[residue_num-1][residue_num-1+index] += w1 * abs(energy)
                    for i, value in enumerate(adjacency_matrix[residue_num-1]):
                        if value != 0:
                            adjacency_matrix[residue_num-1][i] += w2 * abs(energy)
                    for i, row in enumerate(adjacency_matrix):
                        if row[residue_num-1+index] != 0:
                            adjacency_matrix[i][residue_num-1+index] += w2 * abs(energy)
        except IndexError:
            print(f"Error updating adjacency matrix for protein {protein_id}, residue {residue_num}.")
            continue
    return adjacency_matrix

def h_bonds_process(adjacency_matrix_file, dssp_features_file, output_file, w1, w2):
    with open(adjacency_matrix_file, 'rb') as file:
        adjacency_matrices = pickle.load(file)
    with open(dssp_features_file, 'rb') as file:
        dssp_features = pickle.load(file)

    for protein_id, adjacency_matrix in adjacency_matrices.items():
        protein_features = dssp_features.get(protein_id)
        if protein_features is not None:
            adjacency_matrix = update_adjacency_matrix(adjacency_matrix, protein_features, w1, w2, protein_id)
            adjacency_matrices[protein_id] = adjacency_matrix
    with open(output_file, 'wb') as file:
        pickle.dump(adjacency_matrices, file)


if __name__ == '__main__':
    
    # —————————————————— LPE ——————————————————
    pdb_dir = './test_lib/EsmFold/train_set'
    pdb_files = os.listdir(pdb_dir)

    dismatrix_dict = {}
    for pdb_file in pdb_files:
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            dismatrix, edge_index = get_edge_index(pdb_path)
            id = os.path.splitext(pdb_file)[0]
            dismatrix_dict[id] = dismatrix.astype(np.int16)

    with open('./data/E.coli/nesg_all/nesg_train_all_dismatrix_dict_raw.pkl', 'wb') as f:
        pickle.dump(dismatrix_dict, f)
    
    process_adj_matrices('./case study/graph/B5L6K6_dismatrix_dict_row.pkl',
                      './case study/graph/B5L6K6_laplacian_dict.pkl')
    
    # —————————————————— DSSP ——————————————————
    pdb_dir = '/home/dell/disks/xujia/Paper/MyPaper/paper1/test_lib/EsmFold/case_study/B5L6K6/'
    dssp_dir = '/home/dell/disks/xujia/Paper/MyPaper/paper1/test_lib/DSSP/case_study/B5L6K6/'

    for filename in os.listdir(pdb_dir):
        if filename.endswith('.pdb'):
            basename = os.path.splitext(filename)[0]
            cmd = f'dssp -i {pdb_dir}/{filename} -o {dssp_dir}/{basename}.dssp'
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                print(f'Error processing file: {filename}')
                continue

    features = {}

    for filename in os.listdir(dssp_dir):
        if filename.endswith('.dssp'):
            protein_id = os.path.splitext(filename)[0]
            features[protein_id] = {}
            with open(os.path.join(dssp_dir, filename), 'r') as file:
                lines = file.readlines()
                start = next(i for i, line in enumerate(lines) if '#  RESIDUE AA STRUCTURE' in line)
                for line in lines[start+1:]:
                    residue_aa = line[13]
                    if residue_aa == '!':
                        continue
                    residue_num = int(line[5:10])
                    structure = line[16]
                    bp1 = int(line[25:29])
                    bp2 = int(line[29:33])
                    acc = float(line[34:38])
                    nh_o_1 = int(line[39:45]), float(line[46:50])
                    o_hn_1 = int(line[50:56]), float(line[57:61])
                    nh_o_2 = int(line[61:67]), float(line[68:72])
                    o_hn_2 = int(line[72:78]), float(line[79:83])
                    tco = float(line[85:91])
                    kappa = float(line[91:97])
                    alpha = float(line[97:103])
                    phi = float(line[103:109])
                    psi = float(line[109:115])
                    x_ca = float(line[115:122])
                    y_ca = float(line[122:129])
                    z_ca = float(line[129:136])
                    features[protein_id][residue_num] = {
                        'residue_num': residue_num, 
                        'residue_aa': residue_aa,
                        'structure': structure,
                        'bp1': bp1,
                        'bp2': bp2,
                        'acc': acc,
                        'nh_o_1': nh_o_1,
                        'o_hn_1': o_hn_1,
                        'nh_o_2': nh_o_2,
                        'o_hn_2': o_hn_2,
                        'tco': tco,
                        'kappa': kappa,
                        'alpha': alpha,
                        'phi': phi,
                        'psi': psi,
                        'x_ca': x_ca,
                        'y_ca': y_ca,
                        'z_ca': z_ca,
                    }
                    
    with open('./data/soluprot+epsol/soluprot_dssp.pkl', 'wb') as file:
        pickle.dump(features, file)
    
    # —————————————————— Node fea ——————————————————
    with open('./data/soluprot+epsol/soluprot_dssp.pkl', 'rb') as file:
        data = pickle.load(file)

    fasta_file = './data/soluprot+epsol/SoluProt_fea_new.fasta'
    output_file = './data/soluprot+epsol/soluprot_node_fea.pkl'
    node_fea_process(fasta_file, output_file, data)

    # —————————————————— H-bond ——————————————————
    adjacency_matrix_file = './case study/graph/B5L6K6_laplacian_dict.pkl'
    dssp_features_file = './case study/graph/B5L6K6_dssp.pkl'
    output_file = './case study/graph/B5L6K6_LPE_head_5_1.pkl'
    w1 = 5  
    w2 = 1  
    h_bonds_process(adjacency_matrix_file, dssp_features_file, output_file, w1, w2)
    
    