epochs = 250
batch_size = 32
start_epoch = 0

lr = 0.00005 

seed = 2333
num_workers = 4

weight_decay = 0.0001  
gpu = 0  

train_dataset_path = './data/noise_free/eSOL_train_reg.fasta'
test_dataset_path = './data/noise_free/eSOL_test_reg.fasta'

protein2id = './data/noise_free/eSOL_go/train_protein2ind.txt'
protein2ont = './data/noise_free/eSOL_go/train_total.txt'

max_pad_len = 200  

model_path = './Most accurate models/SolLRT_eSOL/reg_total_400_0.211_0.564.pth'
last_model_path = './Most accurate models/last_model.pth'
save_path = './lib_output/noise_free_reg/'

