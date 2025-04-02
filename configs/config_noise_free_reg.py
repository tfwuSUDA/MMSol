epochs = 600
batch_size = 32
start_epoch = 0

lr = 0.00005 

seed = 2333
num_workers = 4

weight_decay = 0.0001  
gpu = 0  

train_dataset_path = './data/noise_free/noise_free_train_reg.fasta'
test_dataset_path = './data/noise_free/noise_free_test_reg.fasta'

protein2id = './data/noise_free/noise_free_go/train_protein2ind.txt'
protein2ont = './data/noise_free/noise_free_go/train_total.txt'

max_pad_len = 200  

model_path = './Best_ckpt/noise_free/reg.pth'
save_path = './output/noise_free_reg/'

