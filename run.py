import numpy as np
from sklearn.model_selection import KFold
from data_runner import data_provider
from torch.utils.data import DataLoader
import torch
import os
import importlib
import inspect

import torch.nn as nn
from torch.utils.data import Subset

import pandas as pd
import metrics as met
import csv
import warnings
import random
import argparse

from util import get_param_combinations,get_model,save_metric,save_result,plot_k_folf,forward_and_loss,save_mf

warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--pred_len", type=int, default=1)
parser.add_argument("--model_name", type=str, default="KAN")

args = parser.parse_args()

# 训练模型
config={
    'data_root':"./dataset",
    'dataset_name':"hanjiang",
    'hist_len':10,
    'pred_len':7,
    'var_num':73,
    'data_split':[8640+2880, 0, 2880],
    "k":5,
    "seed":1,
    "batch_size":64,
    "num_workers":2,
    "learning_rate":1e-4,

    "optimizer_betas":(0.9, 0.999),
    "optimizer_weight_decay":1e-5,
    "dropout":0.2,


    'model_name':"DPL_KAFormer",  # 模型名称
    "epochs":15,

    #快速调试
    "kf":False, #k-flod
    "tt":True, #train and test
    "bpn":4  #best paramter noi

}

config["pred_len"]=args.pred_len
config["model_name"]=args.model_name

print(config)


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 在代码开始时调用
set_seed(config["seed"])


# 参数搜索空间
param_combinations = get_param_combinations(config)

#归一化相关
_norm="std"
_stat = np.load(os.path.join(config['data_root'], config["dataset_name"], 'var_scaler_info.npz'))
_mean=torch.tensor(_stat['mean']).float().to(device)
_std=torch.tensor(_stat['std']).float().to(device)
_min=torch.tensor(_stat['min']).float().to(device)
_max=torch.tensor(_stat['max']).float().to(device)

def inverse_transform_var(data):
    if _norm=="minmax":
        return _min+(data+1.0)*(_max-_min)/2.0
    else: #"norm"
        return (data * _std) + _mean

def transform_var(data):  
    if _norm=="minmax":
        return -1.0+2*(data-_min)/(_max-_min)
    else:
        return (data - _mean) / _std
    

#损失函数
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def get_precipitation_and_attributes(model_name, var_x, var_y):
    df=pd.read_csv("/home/glotm/code/kf/dataset/hanjiang/feature.csv").iloc[:,1:].columns

    var_all=torch.cat([var_x,var_y],dim=1).to(var_x.device)

    prec_idx=[8, 17, 26, 35, 44, 53, 62, 71]
    we=torch.tensor([0.06770909,0.111653249,0.130570316,0.148518791,0.140389194,0.072451036,0.131889334,0.19681899],dtype=var_x.dtype,device=var_x.device)
    ares=39833680857

    attr_idx=[0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56, 63, 64, 65]

    prec= var_all[:, :,prec_idx] * 0.001 * we * ares / 60 / 60 / 24
    # print(prec)
    attributes = transform_var(var_all)[:,:,attr_idx]

    

    return (prec, attributes)



class K_Fold_Module:
    def __init__(self,config, **kargs): 
        self.dropout = config["dropout"]
        self.seed = config["seed"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.learning_rate = config["learning_rate"]
        self.epochs=config["epochs"]
        self.optimizer_betas=config["optimizer_betas"]
        self.optimizer_weight_decay=config["optimizer_weight_decay"]
        self.hist_len=config["hist_len"]
        self.pred_len=config["pred_len"]
        self.var_num=config["var_num"]
        self.model_name=config["model_name"]

        self.num_layer = kargs.get('num_layer', 2)
        self.hidden_num = kargs.get('hidden_num', 16)
        self.num_frequencies = kargs.get('num_frequencies', 5)
        self.d_model = kargs.get('d_model', 256)
        self.e_layer = kargs.get('e_layer', 2)
        self.n_heads = kargs.get('n_heads', 4)
        self.cnn_dim=kargs.get('cnn_dim',4)


        self.criterion=RMSELoss()
        

        self.model=get_model(model_name=config["model_name"],
                             hist_len=self.hist_len,
                             pred_len=self.pred_len,
                             var_num=self.var_num,
                             num_layer=self.num_layer,
                             hidden_num=self.hidden_num,
                             dropout=self.dropout,
                             device=device,
                            seed=self.seed,
                            num_frequencies=self.num_frequencies,
                            d_model=self.d_model,
                            n_heads=self.n_heads,
                            e_layer=self.e_layer,
                            d_layer=self.e_layer,
                            group_num=4,
                            cnn_dim=self.cnn_dim
                             )
        
        self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                betas=self.optimizer_betas, 
                weight_decay=self.optimizer_weight_decay)



    def fit(self,train_set,train_idx, val_idx):
        # train_set=data_provider(config=config, mode='train')

        train_data=DataLoader(Subset(train_set,train_idx), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,drop_last=True)
        val_data=DataLoader(Subset(train_set,val_idx), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,drop_last=True)
        # print(len(train_set),len(train_data))
        # print(len(vaild_set),len(val_data))

        # 训练过程
        train_epoch_ts_loss=[]
        train_epoch_loss=[]
        vaild_epoch_loss=[]
        for epoch in range(self.epochs):  # 假设训练 10 个 epoch
            # Train
            train_preds=[]
            train_labels=[]
            train_ts_preds=[]
            train_ts_labels=[]

            for batch in train_data:
                var_x, marker_x, var_y, marker_y = [x.to(device).float() for x in batch]
                precipitation,attributes=get_precipitation_and_attributes(self.model_name,var_x,var_y)


                self.optimizer.zero_grad()

                #归一化
                inputs = transform_var(var_x)
                targets = transform_var(var_y)
                future_input=targets[...,:-1]

                outputs,loss,_yf,_yn,m_f=forward_and_loss(self.model_name, self.model, self.criterion, inputs, targets, future_input, epoch,True,precipitation,attributes)

                train_ts_preds.append(outputs[:,-1,-1])
                train_ts_labels.append(targets[:,-1,-1])

                loss.backward()
                self.optimizer.step()

                outputs=inverse_transform_var(outputs)
                targets=inverse_transform_var(targets)

                train_preds.append(outputs[:,-1,-1])
                train_labels.append(targets[:,-1,-1])

            # 归一化训练损失
            train_ts_preds = torch.cat(train_ts_preds, dim=0)
            train_ts_labels = torch.cat(train_ts_labels, dim=0)
            train_loss=self.criterion(train_ts_preds, train_ts_labels)
            train_epoch_ts_loss.append(train_loss.item())

            # 反归一化训练损失
            train_preds = torch.cat(train_preds, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            # print(train_preds.size(),train_labels.size())
            rinv_train_loss=self.criterion(train_preds, train_labels)
            train_epoch_loss.append(rinv_train_loss.item())


            #Vaild
            vaild_preds=[]
            vaild_labels=[]
            with torch.no_grad():
                for batch in val_data:
                    var_x, marker_x, var_y, marker_y = [x.to(device).float() for x in batch]

                    precipitation,attributes=get_precipitation_and_attributes(self.model_name,var_x,var_y)

                    #归一化
                    inputs = transform_var(var_x)
                    targets = transform_var(var_y)
                    future_input=targets[...,:-1]

                    outputs,_l,_yf,_yn,m_f=forward_and_loss(self.model_name, self.model, self.criterion, inputs, targets, future_input, epoch,False,precipitation,attributes)

                    outputs=inverse_transform_var(outputs)
                    targets=inverse_transform_var(targets)


                    vaild_preds.append(outputs[:,-1,-1])
                    vaild_labels.append(targets[:,-1,-1])

                vaild_preds = torch.cat(vaild_preds, dim=0)
                vaild_labels = torch.cat(vaild_labels, dim=0)

                rinv_vaild_loss=self.criterion(vaild_preds, vaild_labels)
                vaild_epoch_loss.append(rinv_vaild_loss.item())
           
            if (epoch+1)%5==0:
                print(f"Epoch {epoch+1}/{self.epochs},ts train Loss: {train_epoch_ts_loss[-1]:.3f},train Loss: {train_epoch_loss[-1]:.0f}, vaild Loss: {vaild_epoch_loss[-1]:.0f}")
        
        return train_epoch_loss,vaild_epoch_loss,vaild_preds,vaild_labels
    




class Final_Train_Test:
    def __init__(self,config, **kargs): 
        self.dropout = config["dropout"]
        self.seed = config["seed"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.learning_rate = config["learning_rate"]
        self.epochs=config["epochs"]
        self.optimizer_betas=config["optimizer_betas"]
        self.optimizer_weight_decay=config["optimizer_weight_decay"]
        self.hist_len=config["hist_len"]
        self.pred_len=config["pred_len"]
        self.var_num=config["var_num"]
        
        self.model_name=config["model_name"]


        self.num_layer = kargs.get('num_layer', 2)
        self.hidden_num = kargs.get('hidden_num', 16)
        self.num_frequencies= kargs.get('num_frequencies', 5)
        self.d_model = kargs.get('d_model', 256)
        self.e_layer = kargs.get('e_layer', 2)
        self.n_heads = kargs.get('n_heads', 4)
        self.cnn_dim=kargs.get('cnn_dim',4)


        self.criterion=RMSELoss()
        
        self.model=get_model(model_name=config["model_name"],
                                hist_len=self.hist_len,
                                pred_len=self.pred_len,
                                var_num=self.var_num,
                                num_layer=self.num_layer,
                                hidden_num=self.hidden_num,
                                dropout=self.dropout,
                                device=device,
                                seed=self.seed,
                                num_frequencies=self.num_frequencies,
                                d_model=self.d_model,
                                n_heads=self.n_heads,
                                e_layer=self.e_layer,
                                d_layer=self.e_layer,
                                group_num=4,
                                cnn_dim=self.cnn_dim
                                )

        self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                betas=self.optimizer_betas, 
                weight_decay=self.optimizer_weight_decay)




    def fit(self,train_set,test_set):
        train_data=DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,drop_last=True)
        test_data=DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,drop_last=True)

        # 训练过程
        train_epoch_loss=[]
        test_epoch_loss=[]
        for epoch in range(self.epochs):  # 假设训练 10 个 epoch
            # Train
            train_preds=[]
            train_labels=[]

            for batch in train_data:
                var_x, marker_x, var_y, marker_y = [x.to(device).float() for x in batch]
                self.optimizer.zero_grad()

                precipitation,attributes=get_precipitation_and_attributes(self.model_name,var_x,var_y)

                #归一化
                inputs = transform_var(var_x)
                targets = transform_var(var_y)
                future_input=targets[...,:-1]

                outputs,loss,_yf,_yn,m_f=forward_and_loss(self.model_name, self.model, self.criterion, inputs, targets, future_input, epoch,True,precipitation,attributes)
                
            
                loss.backward()
                self.optimizer.step()

                outputs=inverse_transform_var(outputs)
                targets=inverse_transform_var(targets)

                train_preds.append(outputs[:,-1,-1])
                train_labels.append(targets[:,-1,-1])

            # 反归一化训练损失
            train_preds = torch.cat(train_preds, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            rinv_train_loss=self.criterion(train_preds, train_labels)
            train_epoch_loss.append(rinv_train_loss.item())


            #Test
            test_preds=[]
            test_labels=[]
            test_mf=[]
            test_yf=[]
            test_yn=[]
            with torch.no_grad():
                for batch in test_data:
                    var_x, marker_x, var_y, marker_y = [x.to(device).float() for x in batch]

                    precipitation,attributes=get_precipitation_and_attributes(self.model_name,var_x,var_y)

                    #归一化
                    inputs = transform_var(var_x)
                    targets = transform_var(var_y)
                    future_input=targets[...,:-1]

                    outputs,_l,_yf,_yn,_mf=forward_and_loss(self.model_name, self.model, self.criterion, inputs, targets, future_input, epoch,False,precipitation,attributes,output_mf=True)

                    outputs=inverse_transform_var(outputs)
                    targets=inverse_transform_var(targets)


                    test_preds.append(outputs[:,-1,-1])
                    test_labels.append(targets[:,-1,-1])

                    # print(mf.shape)
                    if _mf is not None:
                        test_mf.append(_mf[:,-1])
                        test_yf.append(_yf[:,-1])
                        test_yn.append(_yn[:,-1])

                test_preds = torch.cat(test_preds, dim=0)
                test_labels = torch.cat(test_labels, dim=0)

                if _mf is not None:
                    test_mf = torch.cat(test_mf, dim=0)
                    test_yf = torch.cat(test_yf, dim=0)
                    test_yn = torch.cat(test_yn, dim=0)
                # print(test_mf.shape)

                rinv_test_loss=self.criterion(test_preds, test_labels)
                test_epoch_loss.append(rinv_test_loss.item())
           
            
            
            print(f"Epoch {epoch+1}/{self.epochs},train Loss: {train_epoch_loss[-1]:.0f}, test Loss: {test_epoch_loss[-1]:.0f}")
        
            if epoch==self.epochs-1:
                save_result(self.pred_len,self.model_name,train_preds,train_labels,test_preds,test_labels)
                save_metric(self.pred_len,self.model_name,train_preds,train_labels,test_preds,test_labels)

                # print(mf)
                save_mf(self.pred_len,self.model_name,test_yf,test_yn,test_mf)

        
        print("Finish!!!!!")
        # return train_epoch_loss,test_epoch_loss,test_preds,test_labels


# 创建保存模型结果的文件夹    
model_result_folder=os.path.join(f"./save_{str(config['pred_len'])}",config["model_name"])
if not os.path.exists(model_result_folder):
    os.makedirs(model_result_folder)


# 假设已有数据
train_data=data_provider(config=config, mode='train')
test_data=data_provider(config=config, mode='test')


# K折交叉验证
kf = KFold(n_splits=config["k"], shuffle=True, random_state=42)

best_score = np.inf
best_params = None

#-------------------
#K-Fold
if config["kf"]==True:
    nps=0   
    for params in param_combinations:
        nps+=1
        fold_accuracies = []
        nk=0



        train_loss_list=[]
        vaild_loss_list=[]

        for train_idx, val_idx in kf.split(np.arange(len(train_data))):
            nk+=1
            model = K_Fold_Module(config,**params)

            print(f"nps: {nps}, Fold: {nk}/{config['k']}, Params: {params}")
            train_epoch_loss,vaild_epoch_loss,vaild_preds,vaild_labels=model.fit(train_data,train_idx,val_idx)

            train_loss_list.append(train_epoch_loss)
            vaild_loss_list.append(vaild_epoch_loss)



            # plt.plot(epochs, train_epoch_loss, color='blue',linewidth=1)
            # plt.plot(epochs, vaild_epoch_loss, color='black',linewidth=1)
        
            fold_accuracies.append(vaild_epoch_loss[-1])  # 使用最后一个epoch的验证损失作为准确率
        
        # 计算每折的平均准确率
        mean_acc = np.mean(fold_accuracies)
        if mean_acc < best_score:
            best_score = mean_acc
            best_params = params


        # plot_k_folf(config["model_name"],nps,train_loss_list,vaild_loss_list)




    print(f"Best Params: {best_params}, Best Score: {best_score}")

    #---------------------------------

    # 保存最佳指标
    base_para_path=os.path.join(f"./save_{str(config['pred_len'])}",config["model_name"],"best_params.csv")
    with open(base_para_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=best_params.keys())
        writer.writeheader()
        writer.writerow(best_params)

    print(f"最佳参数已保存到 {base_para_path}")


#--------------------------------
# 最终训练和测试
if config["kf"]==False:
    best_params=param_combinations[config["bpn"]] 

if config["tt"]==True:
    model=Final_Train_Test(config,**best_params)
    model.fit(train_data,test_data)

#--------------------------------