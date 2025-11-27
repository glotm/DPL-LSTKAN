import torch
from torch import nn
import math

class LSTMLayer(nn.Module):

    def __init__(self,input_len,hidden_num):
        super(LSTMLayer,self).__init__()

        self.input_size = input_len
        self.hidden_num=hidden_num
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_len, hidden_num))
        self.V_i = nn.Parameter(torch.Tensor(hidden_num, hidden_num))
        self.b_i = nn.Parameter(torch.Tensor(hidden_num))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(input_len, hidden_num))
        self.V_f = nn.Parameter(torch.Tensor(hidden_num, hidden_num))
        self.b_f = nn.Parameter(torch.Tensor(hidden_num))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(input_len, hidden_num))
        self.V_c = nn.Parameter(torch.Tensor(hidden_num, hidden_num))
        self.b_c = nn.Parameter(torch.Tensor(hidden_num))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(input_len, hidden_num))
        self.V_o = nn.Parameter(torch.Tensor(hidden_num, hidden_num))
        self.b_o = nn.Parameter(torch.Tensor(hidden_num))
        
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_num)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



    def forward(self,x):
        """
        assumes x.shape represents (sequence_size, batch_size , input_size)
        """
        
        seq_sz,bs, _ = x.size()
        hidden_seq = []

        h_t, c_t = (
            torch.zeros(bs, self.hidden_num).to(x.device),
            torch.zeros(bs, self.hidden_num).to(x.device),
        )
        
        for t in range(seq_sz):
            x_t = x[t, :, :] #[B,C]
            
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0).float().to(x.device) #[L,B,C]
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        hidden_seq = hidden_seq.contiguous()
        return hidden_seq, (h_t, c_t)