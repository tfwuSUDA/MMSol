from Bio.SeqUtils import ProtParam
from Bio import SeqIO
import numpy as np
import csv

kd = {'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
      'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
      'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
      'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2}
IUPACProtein_letters='ACDEFGHIKLMNPQRSTVWY'
def check_seq(seq):
    for aa in seq:
        if aa not in IUPACProtein_letters:
            return False
    return True

features = {"sid":[],"label":[], "label_noise":[],
                "fracnumcharge": [], "aa_turn":[],
                "aa_turn": [], "aa_turn":[], "aa_turn": [],
                "molecular_weight": [], "length": [],
                'avg_molecular_weight': [], "aromaticity": [],
                "instability_index": [], "flexibility": [],
                "gravy": [], "isoelectric_point": [],}

def fracnumcharge(aa_freq):
    return aa_freq['R'] + aa_freq['K'] + aa_freq['D'] + aa_freq['E']

def thermostability(aa_freq):
    return aa_freq['I'] + aa_freq['V'] + aa_freq['Y'] + aa_freq['W'] + \
           aa_freq['R'] + aa_freq['E'] + aa_freq['L']

def aa_turn(aa_freq):
    if aa_freq['R'] == 0:
        return np.nan
    else:
        return aa_freq['K']/aa_freq['R']

def de_mul(aa_freq):
    return aa_freq['D']*aa_freq['E']

standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
def biopython_12(input_file=None):
    with open('example.csv', "w",newline="") as csv_out:
        csv_wr = csv.DictWriter(csv_out, fieldnames=features)
        csv_wr.writeheader()
        for seq in SeqIO.parse('.\data\noise_free\eSOL_train_cls.fasta', "fasta"):
            row = dict.fromkeys(features)  
            description_parts = seq.description.split()
            row['sid'] = description_parts[0][1:]  
            for part in description_parts[1:]:
                if part.startswith('label='):
                    row['label'] = int(part.split('=')[1])
                elif part.startswith('label_noise='):
                    row['label_noise'] = int(part.split('=')[1])
            sequence = str(seq.seq)
            if set(sequence).issubset(standard_amino_acids):
                analysis = ProtParam.ProteinAnalysis(sequence)
                aa_freq = analysis.get_amino_acids_percent()
                
                row['fracnumcharge'] = fracnumcharge(aa_freq)  
                row['length'] = analysis.length  
                if 'R' in aa_freq and 'K' in aa_freq and 'D' in aa_freq and 'E' in aa_freq:
                    row['aa_turn'] = aa_turn(aa_freq) 
                else:
                    row['aa_turn'] = np.nan
                row['molecular_weight'] = analysis.molecular_weight()  
                
                row['avg_molecular_weight'] = row['molecular_weight']/row['length']  
                row['aromaticity'] = analysis.aromaticity()  
                row['instability_index'] = analysis.instability_index() 
                row['flexibility'] = np.mean(analysis.flexibility())  
                row['gravy'] = analysis.gravy()  
                row['isoelectric_point'] = analysis.isoelectric_point()  
                csv_wr.writerow(row)
            else:
                print(f"Warning: Sequence {seq.id} contains non-standard amino acids. Skipping this sequence.")
            
def scale(x):
    import pandas as pd
    import numpy as np
    df1=pd.read_csv("./data/"+str(x),index_col=None,header=None).to_numpy()
    scale_ori={}
    for i in range(len(df1)):
        s=df1[i][0].split(": ")
        scale_ori[str(s[0])]=s[1].strip("  ")
    scale_aa = ['A', 'R', 'N', 'D', 'C','Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P','S', 'T', 'W', 'Y', 'V']
    scale_new={}

    from Bio.SeqUtils import seq3
    for x in scale_aa:
        scale_new[x] = float(scale_ori[seq3(x)])
    return scale_new

def physico_chemical(fa_path, f_csv,scale_name,w,xx):
    aa_scale=scale(scale_name)
    #print(aa_scale)
    features = {"sid":[],
                "label":[], "label_noise":[],
                }
    for window in [xx]:
        features[str(window)]=[]
        
    with open("./features/biofea/"+f_csv, "w",newline="") as csv_out:
        csv_wr = csv.DictWriter(csv_out, fieldnames=features)
        csv_wr.writeheader()
        for seq in SeqIO.parse(fa_path, "fasta"):
            if not check_seq(seq.seq):
                continue
            row = dict.fromkeys(features)
            description_parts = seq.description.split()
            row['sid'] = description_parts[0][1:]  
            for part in description_parts[1:]:
                if part.startswith('label='):
                    row['label'] = int(part.split('=')[1])
                elif part.startswith('label_noise='):
                    row['label_noise'] = int(part.split('=')[1])
            if set(str(seq.seq)).issubset(standard_amino_acids):
                analysis = ProtParam.ProteinAnalysis(str(seq.seq))
                    
                f_scale = np.mean(analysis.protein_scale(aa_scale,w))
                row[str(xx)] = f_scale
                csv_wr.writerow(row)

def scale_main(input_file):
    i=0
    scalew=[3,7,5,]
    for x in ['14_hc.txt','38_hj.txt','43_hh.txt']:
        #print(x)
        xx=x.split("_")[0]
        w=scalew[i]
        physico_chemical("./sequence/"+str(input_file), xx+".csv",x,w,xx)
        i=i+1

biopython_12()
scale_main("eSOL_train_cls.fasta")