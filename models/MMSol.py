import numpy as np
from scipy import sparse
import torch
from torch import nn
from transformers import AutoModel
import pandas as pd
from SparseGO.models.SparseGO import SparseLinearNew as SparseGO

# Pretrain path
PRETRAIN_MODEL_PATH = './models/Protein_LLM/esm2_t6_8M_UR50D'


def create_index(array):
    unique_array = pd.unique(array)

    index = {}
    for i, element in enumerate(unique_array):
        index[element] = i

    return index

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias, seq_mask, graph_mask, subsq_mask = None, valid=False,check=False):
        q = q.float()
        k = k.float()
        v = v.float()   
        orig_q_size = q.size()  

        d_k = self.att_size 
        d_v = self.att_size 
        batch_size = q.size(0) 

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)  
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  
        v = v.transpose(1, 2)                  
        k = k.transpose(1, 2).transpose(2, 3)  
        q = q * self.scale  

        x = torch.matmul(q, k) 
        
        if attn_bias is not None:
            
            attn_bias = attn_bias.masked_fill(graph_mask == 0, -1e9)
            x = x + attn_bias
        if subsq_mask is not None:
            x = x + subsq_mask
        
        if graph_mask is not None:
            x = x.masked_fill(graph_mask == 0, -1e9)

        x = torch.softmax(x, dim=-1)

        if not valid:
            x = self.att_dropout(x)

        x = x.matmul(v)  

        x = x.transpose(1, 2).contiguous() 
        x = x.view(batch_size, -1, self.head_size * d_v) 

        x = self.output_layer(x) 

        assert x.size() == orig_q_size
        return x


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):  
        super(Attention, self).__init__()
        self.input_dim = input_dim  
        self.dense_dim = dense_dim  
        self.n_heads = n_heads 
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)


    def softmax(self, input, axis=1):  
        input_size = input.size()  
        trans_input = input.transpose(axis, len(input_size) - 1)  
        trans_size = trans_input.size()  
        input_2d = trans_input.contiguous().view(-1, trans_size[-1]) 
        soft_max_2d = torch.softmax(input_2d, dim=1)  
        soft_max_nd = soft_max_2d.view(*trans_size) 
        return soft_max_nd.transpose(axis, len(input_size) - 1) 

    def forward(self, input):               
        x = torch.tanh(self.fc1(input))   
        x = self.fc2(x)                    
        x = self.softmax(x, 1)             
        attention = x.transpose(1, 2)      
        return attention



class EncoderLayer(nn.Module):
    
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-5, dtype=torch.float32)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.head_size = head_size
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-5, dtype=torch.float32)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias, seq_mask, graph_mask, valid=False):

        x = x.float()
        y = self.self_attention_norm(x)

        attn_bias = attn_bias.unsqueeze(1)  
        attn_bias = attn_bias.repeat(1, self.head_size, 1, 1) 

        graph_mask = graph_mask.unsqueeze(1) 
        graph_mask = graph_mask.repeat(1, self.head_size, 1, 1)  

        y= self.self_attention(y, y, y, attn_bias, seq_mask, graph_mask,valid=True)  
        if not valid:
            y = self.self_attention_dropout(y)  
        
        x = x + y  

        y = self.ffn_norm(x) 
        y = self.ffn(y)  
        if not valid:
            y = self.ffn_dropout(y)
        x = x + y  

        return x
    

class Model(nn.Module):
    
    def __init__(self, hidden_size=36, ffn_size=256, dropout_rate=0.1, attention_dropout_rate=0.1,
                  head_size=8, num_layers=6, num_labels=2, pretrained_model_path=PRETRAIN_MODEL_PATH,
                  num_neurons_per_GO=1, num_neurons_per_final_GO=1, 
                  num_neurons_final=1, layer_connections=None,
                  p_drop_final=0.3,p_drop_terms=0.3):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(
            pretrained_model_path, num_labels=num_labels, problem_type="single_label_classification"
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size)
                                     for _ in range(num_layers)])
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.num_classes = num_labels

        self.feature_dim = 8 
        self.seq_output_dim = self.model.config.hidden_size  
        
        self.seq_fc = nn.Linear(self.seq_output_dim, 32)
        self.fea_fc = nn.Linear(self.feature_dim, 64)
        self.graph_fc = nn.Linear(36, 32)
        self.graph_attention = Attention(36, 128, 8)
        
        self.node_emb = nn.Linear(36, 36)

        # SparseGO
        self.num_neurons_per_GO = num_neurons_per_GO
        self.num_neurons_per_final_GO = num_neurons_per_final_GO
        self.layer_connections=layer_connections

        # # (1) Layer of genes with terms
        mf_keys = ['GO:0001618', 'GO:0003677', 'GO:0003723', 'GO:0003774', 'GO:0003824', 'GO:0003924', 'GO:0005198', 'GO:0005215', 'GO:0008092', 
           'GO:0008289', 'GO:0009975', 'GO:0016209', 'GO:0016491', 'GO:0016740', 'GO:0016787', 'GO:0016829', 'GO:0016853', 'GO:0016874', 
           'GO:0031386', 'GO:0038024', 'GO:0042393', 'GO:0044183', 'GO:0045182', 'GO:0045735', 'GO:0048018', 'GO:0060089', 'GO:0060090',
           'GO:0090729', 'GO:0098631', 'GO:0098772', 'GO:0120274', 'GO:0140096', 'GO:0140097', 'GO:0140098', 'GO:0140104', 'GO:0140110',
           'GO:0140223', 'GO:0140299', 'GO:0140313', 'GO:0140657', 'GO:0003674']

        bp_keys = ['GO:0000278', 'GO:0000910', 'GO:0002181', 'GO:0002376', 'GO:0003012', 'GO:0003013', 'GO:0003014', 'GO:0003016', 'GO:0005975',
                'GO:0006091', 'GO:0006260', 'GO:0006281', 'GO:0006310', 'GO:0006325', 'GO:0006351', 'GO:0006355', 'GO:0006399', 'GO:0006457',
                'GO:0006486', 'GO:0006520', 'GO:0006575', 'GO:0006629', 'GO:0006766', 'GO:0006790', 'GO:0006886', 'GO:0006913', 'GO:0006914',
                'GO:0006954', 'GO:0007005', 'GO:0007010', 'GO:0007018', 'GO:0007031', 'GO:0007040', 'GO:0007059', 'GO:0007155', 'GO:0007163',
                'GO:0012501', 'GO:0015979', 'GO:0016071', 'GO:0016073', 'GO:0016192', 'GO:0022414', 'GO:0022600', 'GO:0023052', 'GO:0030154',
                'GO:0030163', 'GO:0030198', 'GO:0031047', 'GO:0032200', 'GO:0034330', 'GO:0042060', 'GO:0044782', 'GO:0048856', 'GO:0048870',
                'GO:0050877', 'GO:0050886', 'GO:0051604', 'GO:0055085', 'GO:0055086', 'GO:0061007', 'GO:0061024', 'GO:0065003', 'GO:0071554',
                'GO:0071941', 'GO:0072659', 'GO:0098542', 'GO:0098754', 'GO:0140013', 'GO:0140014', 'GO:0140053', 'GO:1901135', 'GO:0008150']

        cc_keys = ['GO:0000228', 'GO:0005576', 'GO:0005615', 'GO:0005618', 'GO:0005634', 'GO:0005635', 'GO:0005654', 'GO:0005694', 'GO:0005730',
                'GO:0005739', 'GO:0005764', 'GO:0005768', 'GO:0005773', 'GO:0005777', 'GO:0005783', 'GO:0005794', 'GO:0005811', 'GO:0005815',
                'GO:0005829', 'GO:0005840', 'GO:0005856', 'GO:0005886', 'GO:0005929', 'GO:0009536', 'GO:0009579', 'GO:0030312', 'GO:0031012',
                'GO:0031410', 'GO:0043226', 'GO:0005575']
        
        head_key = ['GO:0000000']

        all_keys = mf_keys + bp_keys + cc_keys + head_key
        input_id = {go_id: idx for idx, go_id in enumerate(all_keys)}

        for i in range(1,len(layer_connections)):
            if i == len(layer_connections)-1:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_final_GO,p_drop_terms)
            else:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_GO,p_drop_terms)
        self.SparseGO_fc = nn.Linear(29, 32)
        
        self.fusion_layer = nn.Linear(164, 128)
        self.fc = nn.Linear(128, num_labels)
        self.relu = nn.ReLU()  

    def terms_layer(self, input_id, layer_pairs, number,neurons_per_GO,p_drop_terms):
        output_id = create_index(layer_pairs[:,0])

        rows = [output_id[term] for term in layer_pairs[:,0]]
        columns = [input_id[term] for term in layer_pairs[:,1]]  

        data = np.ones(len(rows))

        connections_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(len(output_id), len(input_id)))

        ones = sparse.csr_matrix(np.ones([neurons_per_GO, self.num_neurons_per_GO], dtype = int))
        connections_matrix_more_neurons = sparse.csr_matrix(sparse.kron(connections_matrix, ones))

        rows_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[0]).view(1,-1).long()
        columns_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[1]).view(1,-1).long()
        connections = torch.cat((rows_more_neurons, columns_more_neurons), dim=0)

        input_terms = self.num_neurons_per_GO*len(input_id)
        output_terms = neurons_per_GO*len(output_id)
        # print(input_terms, output_terms)

        self.add_module('GO_terms_sparse_linear_'+number, SparseGO(input_terms, output_terms, connectivity=connections))
        self.add_module('drop_'+number, nn.Dropout(p_drop_terms))
        self.add_module('GO_terms_tanh_'+number, nn.Tanh())
        self.add_module('GO_terms_batchnorm_'+number, nn.BatchNorm1d(input_terms))
        return output_id



    def forward(self, input_ids, attention_mask, feature_vector, GO_fea, x, attn_bias, seq_mask, graph_mask, valid=False):
        # graph
        x = x.float()
        x = self.node_emb(x)
        x = self.relu(x)
        seq_mask = self.node_emb(seq_mask)
        seq_mask = self.relu(seq_mask)
        for layer in self.layers:
            x= layer(x, attn_bias, seq_mask, graph_mask, valid)
        att = self.graph_attention(x)                                             
        node_feature_embedding = att @ x                                    
        x = torch.sum(node_feature_embedding,1) / self.graph_attention.n_heads  
        x = x.squeeze(dim=1) 
        graph_output = x + self.lm_output_learned_bias

        # seq
        esm_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = esm_outputs.last_hidden_state[:, 0, :]  
        seq_output = self.seq_fc(seq_output)
        seq_output = self.relu(seq_output)

        # GO
        terms_output = GO_fea.repeat_interleave(1, dim=-1)
        layers = []
        for i in range(1, len(self.layer_connections)):
            terms_output = self._modules['drop_'+str(i)](terms_output)
            terms_output = self._modules['GO_terms_tanh_'+str(i)](self._modules['GO_terms_sparse_linear_'+str(i)](terms_output))
            layers.append(terms_output)  
        terms_output = torch.cat(layers, dim=1)
        terms_output = self.SparseGO_fc(terms_output)

        # fea
        feature_vector = self.fea_fc(feature_vector)

        # ————concat————
        vec_concat = torch.cat([seq_output, graph_output, feature_vector, terms_output], dim=-1)
        fusion_input = vec_concat
        fusion_input = fusion_input.float()
        fusion_output = self.fusion_layer(fusion_input)
        fusion_output = self.relu(fusion_output)
        fusion_output = self.fc(fusion_output)
        return fusion_output
