import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.3)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for act_fn') 

class BLSTM_frame_sig_att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn):
        super().__init__()
        self.blstm = nn.LSTM(input_size = input_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = dropout, 
                             bidirectional = True, 
                             batch_first = True)
        self.linear1 = nn.Linear(hidden_size*2, linear_output, bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.hasqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=8)
        
        self.hasqiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.hasqiaverage_score = nn.AdaptiveAvgPool1d(1)  
            
    def forward(self, x): #hl:(B,6)
        B, Freq, T = x.size()
        x = x.permute(0,2,1) #(B, 257, T_length)->(B, T_length, 257) 
        
        out, _ = self.blstm(x) #(B,T, 2*hidden)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0,1) #(T_length, B,  128) 
        hasqi, _ = self.hasqiAtt_layer(out,out,out) 
        hasqi = hasqi.transpose(0,1) #(B, T_length, 128)  
        hasqi = self.hasqiframe_score(hasqi) #(B, T_length, 1) 
        hasqi = self.sigmoid(hasqi) #pass a sigmoid
        hasqi_fram = hasqi.permute(0,2,1) #(B, 1, T_length) 
        hasqi_avg = self.hasqiaverage_score(hasqi_fram)  #(B,1,1)
        
        return hasqi_fram, hasqi_avg.squeeze(1) #(B, 1, T_length) (B,1) 
       
