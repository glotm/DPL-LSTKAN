import torch.nn as nn
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
    
class Time_Encoder(nn.Module):
    def __init__(self):
        super(Time_Encoder, self).__init__()

        self.weekday_size = 7
        self.day_size = 32
        self.month_size = 13
        self.yearday_size=367

    
    def forward(self, x):
        #[B,L,N] , N=[Year,Month,Day,YearDay,WeekDay]


        # x[:,:,1]=x[:,:,1]/self.month_size
        # x[:,:,2]=x[:,:,2]/self.day_size
        # x[:,:,3]=x[:,:,3]/self.yearday_size
        # x[:,:,4]=x[:,:,4]/self.weekday_size
        # return x[:,:,-4:]

        x[:,:,1]=x[:,:,1]/self.month_size
        x[:,:,2]=x[:,:,2]/self.day_size
        x[:,:,3]=x[:,:,3]/self.yearday_size
        x[:,:,4]=x[:,:,4]/self.weekday_size

        return torch.concat([x[:,:,1:2],x[:,:,3:4]],dim=-1).float().to(x.device)