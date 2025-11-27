import torch
import torch.nn as nn
from core.kan.MultKAN import MultKAN
from core.kan import plt,nsimplify,ex_round


class Subsequence_Mask():
    def __init__(self, seq_len,pred_len,device):
        self._mask = torch.ones((seq_len,seq_len), dtype=torch.bool).to(device)
        self._mask[:, :-pred_len] = 0

    @property
    def mask(self):
        return self._mask



def normal(shape,device="cuda"):
    return torch.randn(shape) * 0.01


class DPL(nn.Module):
    def __init__(self,k,np,na,device="cuda",stat_inut=None):
        super(DPL, self).__init__()
        self.k = k
        self.np = np
        self.na = na
        self.device = device
        self.lamb=4
        self.stat_input=stat_inut

        
        self.w_rp=nn.Parameter(normal((np,k, k)))  # [Np,K]
        # print(self.w_rp.shape)
        self.w_ra=nn.Parameter(normal((na,k,k))) #[Na,K]
        self.w_rs=nn.Parameter(normal((k,k,k))) #[k,K]
        self.b_r=nn.Parameter(torch.zeros(k,k)) #[k]
 
       

        self.s_max=None
        self.s_fc=None
        if self.stat_inut is not None:
            self.s_max=nn.Linear(len(stat_inut),k)
            self.s_fc=nn.Linear(len(stat_inut),k)
        else:
            self.s_max=nn.Parameter(torch.zeros(k))
            self.s_fc=nn.Parameter(torch.zeros(k))

      
        self.w_ip=nn.Parameter(normal((np,np, k)))  # [Np,K]
        self.w_ia=nn.Parameter(normal((na,np, k))) #[Na,K]
        self.w_is=nn.Parameter(normal((k,np, k))) #[k,K]
        self.b_i=nn.Parameter(torch.zeros(np,k)) #[k]

        
        self.w_op=nn.Parameter(normal((np, k)))  # [Np,K]
        self.w_oa=nn.Parameter(normal((na, k))) #[Na,K]
        self.w_os=nn.Parameter(normal((k,k))) #[k,K]
        self.b_o=nn.Parameter(torch.zeros(k)) #[k]

        params=[self.w_rp, self.w_ra, self.w_rs, self.b_r,
                self.w_ip, self.w_ia, self.w_is, self.b_i,
                self.w_op, self.w_oa, self.w_os, self.b_o,
                self.s_max, self.s_fc]
        
        for p in params:
            p=p.to(self.device)


        #N-CI 参数
        _s=np+na+k
        self.N_CI=nn.Sequential(
            nn.Linear(_s,4*_s),
            nn.ReLU(),
            nn.Linear(4*_s,k),
            nn.ReLU(),
        )



    def forward(self,precipitation,attributes):
        B,L,_=precipitation.shape
        # precipitation: [B,L,Np]  attributes: [B,L,Na]

        s=torch.zeros(B,self.k).to(self.device)  # [B,K]
        s_list=[]
        q_list=[]

        for t in range(L):
            p= precipitation[:,t,:]  # [B,Np]
            a= attributes[:,t,:]  # [B,Na]


            h_s= s / (s.sum(dim=1, keepdim=True) + 1e-8)  # [B,K] 归一化

            
            # print(h_s.shape)
            # print(p.shape, a.shape, h_s.shape)
            r_s= torch.softmax(torch.relu(torch.einsum('bn,nkl->bkl', p, self.w_rp) + #[B,Np] @ [Np,K,K]
                                           torch.einsum('bn,nkl->bkl', a, self.w_ra) + #[B,Na] @ [Na,K,K]
                                           torch.einsum('bn,nkl->bkl', h_s, self.w_rs) + # [B,K] @ [K,K,K]
                                           self.b_r.repeat(B,1,1)
                                           ),dim=-1)   #[B,K,K]
            
            # print(r_s.shape, h_s.shape)
            
            
            s_re=torch.einsum('bkl,bl->bk', r_s,s) #[B,K,K] @ [B,K] = [B,K]

            # print((p * self.w_ip).shape)

          
            i_s= torch.softmax(torch.relu(torch.einsum('bn,nkl->bkl', p, self.w_ip) + #[B,Np] @ [Np,Np,K]
                                           torch.einsum('bn,nkl->bkl', a, self.w_ia) + #[B,Na] @ [Na,Np,K]
                                           torch.einsum('bn,nkl->bkl', h_s, self.w_is) + # [B,K] @ [K,K,K]
                                           self.b_i.repeat(B,1,1)
                                           ),dim=-1)   #[B,Np,K]
            

            o_1= torch.softmax(torch.relu(torch.einsum('bn,nl->bl', p, self.w_op) + #[B,Np] @ [Np,K]
                                torch.einsum('bn,nl->bl', a, self.w_oa) + #[B,Na] @ [Na,K]
                                torch.einsum('bn,nl->bl', h_s,  self.w_os) + # [B,K] @ [K,K]
                                self.b_o.repeat(B,1)),dim=-1)   #[B,K]
            
            m_in=torch.einsum("bn,bnk->bk",p,i_s) #[B,Np] @ [B,Np,K] = [B,K]
            

            _re=None
            _max=None
            q1=None
            q2=None

            if self.stat_inut is not None:
                _max=self.s_max(self.stat_input)
                _fc=self.s_fc(self.stat_input)

                q1=m_in * (torch.tanh((s_re-_max)*self.lamb)+1)/2  #[B,K] * [B,K] = [B,K]
                # print(o_1.shape,torch.min(self.s_fc,self.s_max).shape)
                q2=o_1 * torch.relu(s_re-torch.min(_fc,_max))  #[B,K] * [B,K] = [B,K]
            else:
                q1=m_in * (torch.tanh((s_re-self.s_max)*self.lamb)+1)/2  #[B,K] * [B,K] = [B,K]
                # print(o_1.shape,torch.min(self.s_fc,self.s_max).shape)
                q2=o_1 * torch.relu(s_re-torch.min(self.s_fc,self.s_max))  #[B,K] * [B,K] = [B,K]

            q1=m_in * (torch.tanh((s_re-self.s_max)*self.lamb)+1)/2  #[B,K] * [B,K] = [B,K]
            # print(o_1.shape,torch.min(self.s_fc,self.s_max).shape)
            q2=o_1 * torch.relu(s_re-torch.min(self.s_fc,self.s_max))  #[B,K] * [B,K] = [B,K]

            n_ci=self.N_CI(torch.cat([p,a,h_s],dim=-1))  # [B,Np+Na+K] -> [B,K]


            # print(s.shape,m_in.shape,q1.shape, q2.shape, n_ci.shape)
            s = s + m_in - q1 - q2 - n_ci #[B,K]
            # print(s.shape)
            q=torch.sum((q1+q2),dim=1,keepdim=True)  # [B,1]


            s_list.append(s.unsqueeze(1))
            q_list.append(q.unsqueeze(1))
        
        s_list=torch.concat(s_list,dim=1)  # [B,L,K]
        q_list=torch.concat(q_list,dim=1)  # [B,L,1]
        # print(s_list.shape, q_list.shape)

        return s_list, q_list  # [B,L,K], [B,L,1]
            

class DPL_LSTKAN(nn.Module):
    def __init__(self,hist_len,pred_len,var_num,d_model,e_layer,d_layer,dropout,device="cuda",output_attention=False,activation="relu",d_ff=None,n_heads=8,seed=1):
        super().__init__()
        self.pred_len=pred_len
        self.hist_len=hist_len
        self.var_num=var_num
        
        self.d_model = d_model
        self.src_len=self.hist_len+self.pred_len
        self.src_size=var_num-1
        self.e_layer=e_layer
        self.tgt_size=1
        self.dropout=dropout
        self.device=device
        self.output_attention=output_attention
        self.d_ff=d_ff
        self.activation=activation
        self.n_heads=n_heads
        self.tgt_len=self.hist_len+self.pred_len
        self.d_layer=d_layer
        self.seed=seed

        self.src_pos_emb = nn.Embedding(self.src_len, self.d_model)
        self.tgt_pos_emb = nn.Embedding(self.tgt_len, self.d_model)
        self.src_linear = nn.Linear(self.src_size, self.d_model)
        self.tgt_linear = nn.Linear(self.tgt_size, self.d_model)

        
        self.encoder=nn.LSTM(
            input_size=self.src_size,
            hidden_size=self.d_model,
            num_layers=self.e_layer,
            dropout=self.dropout,
            batch_first=True,
        )




        self.decoder_near = MultKAN(width=[self.d_model+hist_len,self.d_model,self.d_model,self.tgt_size],grid=10,k=3,grid_range=[-1,1],device="cuda",symbolic_enabled=False,seed=seed).speed()
        self.decoder_far = MultKAN(width=[self.d_model+hist_len,self.d_model,self.d_model,self.tgt_size],grid=10,k=3,grid_range=[-1,1],device="cuda",symbolic_enabled=False,seed=seed).speed()

        self.pojo_cls=nn.Linear(2*self.d_model,self.tgt_size)


        dpl_cell=512
        np=8
        na=self.d_model
        self.dpl=DPL(dpl_cell,np,na,device=device) 
        self.merg=nn.Linear(self.pred_len, 1)
        


    def t_forward(self, src, tgt,precipitation,attributes):
        # Position Embedding and Input Projection
        batch_size = src.shape[0]
        src_inputs = src # [B, L+S, N]
        tgt_inputs = tgt[:,:self.hist_len,:] # [B, L+S, 1]

        # Encoder
        enc_self_attn_mask = None
        enc_outputs,_ = self.encoder(src_inputs)
        # enc_outputs: [batch_size, src_len, d_model]


        ka_input = torch.concat([enc_outputs[:,-1,:],tgt_inputs[...,0]],dim=-1) 

        yn = self.decoder_near(ka_input)
        yf=self.decoder_far(ka_input)


        #DPL
        s_list, q_list = self.dpl(precipitation, enc_outputs)  # [B,L,K], [B,L,1]
        yi= self.merg(q_list[:,-self.pred_len:,0])  # [B,1]
        mf=torch.sigmoid(0.2*yi) 


        output=yf*mf+yn*(1-mf) #[B,1]

        output=output.repeat(1,self.pred_len*self.var_num)
        output=output.reshape(-1,self.pred_len,self.var_num)

        return output,yf,yn,mf,q_list,s_list #[B,S,N]
    
    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec,future_input=None,precipitation=None,attributes=None):
        B, _, N = x_enc.shape # B L N

        enc_input=torch.concat([x_enc[...,:-1],future_input],dim=1).to(x_enc.device)
        dec_input=torch.concat([x_enc[...,-1:],torch.zeros(B,self.pred_len,1).to(future_input.device)],dim=1).to(x_enc.device)

        output,yf,yn,mf,pi_q,pi_s=self.t_forward(enc_input, dec_input,precipitation,attributes) #[B,L+S,N] 这里可以不用取后面pred长度，外面会取

        return output,yf,yn,mf,pi_q,pi_s
    

