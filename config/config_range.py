epochs = 1000
batch_size = 32
start_epoch = 0

lr = 0.00005 

seed = 2333
num_workers = 4

weight_decay = 0.0001  
gpu = 0  

train_dataset_path = './data/test_data/merged.fasta'
test_dataset_path_eSOL = './data/noise_free/eSOL_test_reg.fasta'
test_dataset_path_nesg = '/linhaitao/xujia/MMSol/data/test_data/merged_nesg_train_1.fasta'

protein2id = './data/noise_free/eSOL_go/eSOL_train_protein2ind.txt'
protein2ont = './data/noise_free/eSOL_go/eSOL_train_total.txt'

max_pad_len = 200  

model_path = '/linhaitao/xujia/MMSol/Best_ckpt/noise_free_reg/reg_21.3_55.5.pth'
last_model_path = './Most accurate models/last_model.pth'
save_path = './lib_output/range/'


