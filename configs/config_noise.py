epochs = 100
batch_size = 16
start_epoch = 0

lr = 0.00005

seed = 2333
num_workers = 4

weight_decay = 0.001  

gpu = 0  

train_dataset_path = './data/noise/train.fasta'
valid_dataset_path = './data/noise/valid.fasta'
test_dataset_path = './data/noise/test.fasta'
train_dataset_changed_path = './data/noise/noise_train_changed.fasta'

protein2id = './data/noise/noise_go/noise_protein2ind.txt'
protein2ont = './data/noise/noise_go/noise_total.txt'

max_pad_len = 400  

model_path = './Best_ckpt/noise/cls.pth'
save_path = './output/noise_cls/'
