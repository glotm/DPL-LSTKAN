import importlib
import inspect
import os

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import core.metrics as met
import pandas as pd
from sympy.printing.latex import latex
import pickle

class RMSE():
    def __init__(self):
        super().__init__()
        self.n=torch.tensor(0.).to("cuda")
        self.sum=torch.tensor(0.).to("cuda")

    def update(self,preds,target):
        assert preds.shape == target.shape
        self.n+=target.shape[0]
        self.sum+=torch.sum((preds-target)**2)

    def compute(self):
        return torch.sqrt(self.sum/self.n)
    

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


class HanJiangRunner_AddFuture(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss() #MSE，可以修改

        stat = np.load(os.path.join(self.hparams.data_root, self.hparams.dataset_name, 'var_scaler_info.npz'))
    
        self.register_buffer('mean', torch.tensor(stat['mean']).float())
        self.register_buffer('std', torch.tensor(stat['std']).float())

        self.register_buffer('min', torch.tensor(stat['min']).float())
        self.register_buffer('max', torch.tensor(stat['max']).float())




    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch] #在infromer中，vay是有label_len的历史信息的，这里没有，后面需要修改

        #trans std
        var_x=self.transform_var(var_x)
        var_y=self.transform_var(var_y)

 

        #dec input create by concat
        if self.hparams.padding==0:
            dec_inp = torch.zeros([var_y.shape[0], self.hparams.pred_len, var_y.shape[-1]]).float().to(var_x.device)
        elif self.hparams.padding==1:
            dec_inp = torch.ones([var_y.shape[0], self.hparams.pred_len, var_y.shape[-1]]).float().to(var_x.device)

        dec_inp = torch.cat([var_x[:,-self.hparams.label_len:,:], dec_inp], dim=1).float().to(var_x.device)
        dec_marker_inp=torch.cat([marker_x[:,-self.hparams.label_len:,:],marker_y],dim=1).float().to(var_x.device)

        future_input=var_y[...,:-1].to(var_y.device) ##未来气象信息

        label = var_y[:, -self.hparams.pred_len:, :] 

        y_all=torch.concat([var_x[:,-self.hparams.label_len:,-1],var_y[...,-1]],dim=1).to(var_x.device)

        prediction = self.model(var_x, marker_x,dec_inp,dec_marker_inp,future_input)[:, -self.hparams.pred_len:, :] #std
        return prediction, label
    



    #Train
    def on_train_epoch_start(self):
        self.train_preds=torch.tensor([])
        self.train_labels=torch.tensor([])

    def on_train_epoch_end(self):
        return None
    

    # def dpl(self,prediction, label):
    #     #[B,1]
    #     loss=torch.
    #     for 


    def training_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx) #[B,L,N]
        # print(label[:,-1,-1])
        loss = self.loss_function(prediction[:,-1,-1],label[:,-1,-1]) #[B]
        
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)

        self.train_preds=torch.concat([self.train_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.train_labels=torch.concat([self.train_labels,label[:,-1,-1].cpu()],dim=0)

        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # print(f"梯度范数：{total_norm:.2f}")  # 正常范围：0.1-10.0
        # nn.functional.kl_div(pg_l, prob, reduction='batchmean')
        # print(pg_l,prob)
        # print(loss.shape)
        return loss
    
    
    # Vaild
    def on_validation_epoch_start(self):
        self.vaild_preds=torch.tensor([])
        self.vaild_labels=torch.tensor([])
        

    def on_validation_epoch_end(self):
        return None


    def validation_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx)
        
        # loss = self.loss_function(prediction[:,-1,-1],label[:,-1,-1])
        
        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)

        self.vaild_preds=torch.concat([self.vaild_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.vaild_labels=torch.concat([self.vaild_labels,label[:,-1,-1].cpu()],dim=0)


        
        loss=self.loss_function(self.vaild_preds,self.vaild_labels)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss




    #Test
    def on_test_start(self):
        self.test_preds=torch.tensor([])
        self.test_labels=torch.tensor([])
    
    def on_test_end(self):
        save_dir = self.hparams.test_result_path

        data=torch.concat([self.test_preds.reshape(-1,1),self.test_labels.reshape(-1,1)],dim=1)
        print(data.shape)


        #保存测试集结果
        df=pd.DataFrame(data,columns=["preds","labels"])
        df.to_csv(save_dir,index=False,encoding="UTF-8")



        #指标获取
        metric_save_path=self.hparams.metric_save_path
        dict_of_metric={
            "conf_hash":self.hparams.conf_hash,
        }
        header=["mode","NSE","MAE","KGE","RMSE","R"]
        metric_list=[]
        modes=["train","vaild","test"]
        for mode in modes:
            pred=None
            label=None
            if mode=="train":
                pred=self.train_preds
                label=self.train_labels
            elif mode=="vaild":
                pred=self.vaild_preds
                label=self.vaild_labels
            elif mode=="test":
                pred=self.test_preds
                label=self.test_labels
        
            _nse="{:.3f}".format(met.NSE(pred,label).item())
            _mae="{:.0f}".format(met.MAE(pred,label).item())
            _kge="{:.3f}".format(met.KGE(pred,label).item())
            _rmse="{:.0f}".format(met.RMSE(pred,label).item())
            _r="{:.3f}".format(met.R(pred,label).item())

            line=[mode,_nse,_mae,_kge,_rmse,_r]
            metric_list.append(line)
            print(line)

            dict_of_metric[mode+"_NSE"]=_nse
            dict_of_metric[mode+"_MAE"]=_mae
            dict_of_metric[mode+"_KGE"]=_kge
            dict_of_metric[mode+"_rmse"]=_rmse
            dict_of_metric[mode+"_r"]=_r

            df=pd.DataFrame(metric_list,columns=header)
            df.to_csv(metric_save_path,index=False,encoding="UTF-8")

            if mode=="test":
                self.update_metric_in_log(dict_of_metric)

        # np.savez(save_dir,preds=self.test_preds,labels=self.test_labels)


    def test_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx) #[B,L,N]
        
        
        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)


        self.test_preds=torch.concat([self.test_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.test_labels=torch.concat([self.test_labels,label[:,-1,-1].cpu()],dim=0)
        # loss=self.loss_function(self.test_preds,self.test_labels)

        # self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        


        
    def update_metric_in_log(self,metric_dict):
        log_path=self.hparams.log_metric_csv_file

        if not os.path.exists(log_path):
            # 创建一个新的DataFrame并写入文件
            df = pd.DataFrame([metric_dict])
            df.to_csv(log_path, index=False)
        else:
            # 如果文件存在，读取文件并更新或新增数据
            df = pd.read_csv(log_path)
            if metric_dict["conf_hash"] in df['conf_hash'].values:
                # 更新指定ID对应的行
                df.loc[df['conf_hash'] == metric_dict["conf_hash"], metric_dict.keys()] = metric_dict.values()
            else:
                # 如果ID不存在，新增一行数据
                df = pd.concat([df, pd.DataFrame([metric_dict])], ignore_index=True)
            # 保存回CSV文件
            df.to_csv(log_path, index=False)



    def configure_loss(self):
        class RMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                
            def forward(self,yhat,y):
                return torch.sqrt(self.mse(yhat,y))

        class DplLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                
            def forward(self,yhat,y):
                loss=0.
                # print(yhat,y)
                for i in range(yhat.shape[0]):
                    if y[i].item() > 5.:
                        loss+=torch.exp(yhat[i:i+1]-y[i:i+1])-1
                    else:
                        loss+=torch.pow(yhat[i:i+1]-y[i:i+1],2)

                # print(torch.concat(loss,dim=0).shape)
                return loss/yhat.shape[0]


        self.loss_function = RMSELoss()


    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.lr, 
                betas=self.hparams.optimizer_betas, 
                weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(
                self.parameters(), 
                lr=self.hparams.lr, 
                max_iter=self.hparams.lr_max_iter)
        else:
            raise ValueError('Invalid optimizer type!')


        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, 
                    step_size=self.hparams.lr_step_size, 
                    gamma=self.hparams.lr_gamma)
            }
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            lr_scheduler = {
                "scheduler": lrs.MultiStepLR(
                    optimizer, 
                    milestones=self.hparams.milestones, 
                    gamma=self.hparams.gamma)
            }
        elif self.hparams.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=self.hparams.lrs_factor, 
                    patience=self.hparams.lrs_patience),
                "monitor": self.hparams.val_metric
            }
        elif self.hparams.lr_scheduler == 'WSD':
            assert self.hparams.lr_warmup_end_epochs < self.hparams.lr_stable_end_epochs < self.hparams.max_epochs

            def wsd_lr_lambda(epoch):
                if epoch < self.hparams.lr_warmup_end_epochs:
                    return (epoch + 1) / self.hparams.lr_warmup_end_epochs
                if self.hparams.lr_warmup_end_epochs <= epoch < self.hparams.lr_stable_end_epochs:
                    return 1.0
                if self.hparams.lr_stable_end_epochs <= epoch <= self.hparams.max_epochs:
                    return (epoch + 1 - self.hparams.lr_stable_end_epochs) / (
                            self.hparams.max_epochs - self.hparams.lr_stable_end_epochs)

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda),
            }
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }



    def load_model(self):
        model_name = self.hparams.model_name
        # Model = getattr(importlib.import_module('.' + model_name, package='core.model'), model_name)
        Model = getattr(importlib.import_module('.' + model_name, package='core.model'), model_name)
        self.model = self.instancialize(Model)
        

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        model_class_args = inspect.getfullargspec(Model.__init__).args[1:]  # 获取模型参数
        interface_args = self.hparams.keys()
        model_args_instance = {}
        for arg in model_class_args:
            if arg in interface_args:
                model_args_instance[arg] = getattr(self.hparams, arg)
        return Model(**model_args_instance)
    

    # MIN MAX
    def inverse_transform_var(self, data):
        if self.hparams.norm=="minmax":
            return self.min+(data+1.0)*(self.max-self.min)/2.0
        else: #"norm"
            return (data * self.std) + self.mean
    
    def transform_var(self, data):  
        if self.hparams.norm=="minmax":
            return -1.0+2*(data-self.min)/(self.max-self.min)
        else:
            return (data - self.mean) / self.std
    

    









class HanJiangRunner_AddFuture_KF(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss() #MSE，可以修改

        stat = np.load(os.path.join(self.hparams.data_root, self.hparams.dataset_name, 'var_scaler_info.npz'))
    
        self.register_buffer('mean', torch.tensor(stat['mean']).float())
        self.register_buffer('std', torch.tensor(stat['std']).float())

        self.register_buffer('min', torch.tensor(stat['min']).float())
        self.register_buffer('max', torch.tensor(stat['max']).float())




    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch] #在infromer中，vay是有label_len的历史信息的，这里没有，后面需要修改

        #trans std
        var_x=self.transform_var(var_x)
        var_y=self.transform_var(var_y)

 

        #dec input create by concat
        if self.hparams.padding==0:
            dec_inp = torch.zeros([var_y.shape[0], self.hparams.pred_len, var_y.shape[-1]]).float().to(var_x.device)
        elif self.hparams.padding==1:
            dec_inp = torch.ones([var_y.shape[0], self.hparams.pred_len, var_y.shape[-1]]).float().to(var_x.device)

        dec_inp = torch.cat([var_x[:,-self.hparams.label_len:,:], dec_inp], dim=1).float().to(var_x.device)
        dec_marker_inp=torch.cat([marker_x[:,-self.hparams.label_len:,:],marker_y],dim=1).float().to(var_x.device)

        future_input=var_y[...,:-1].to(var_y.device) ##未来气象信息

        label = var_y[:, -self.hparams.pred_len:, :] 

        y_all=torch.concat([var_x[:,-self.hparams.label_len:,-1],var_y[...,-1]],dim=1).to(var_x.device)

        prediction = self.model(var_x, marker_x,dec_inp,dec_marker_inp,future_input)[:, -self.hparams.pred_len:, :] #std
        return prediction, label
    



    #Train
    def on_train_epoch_start(self):
        self.train_preds=torch.tensor([])
        self.train_labels=torch.tensor([])

    def on_train_epoch_end(self):
        return None
    

    # def dpl(self,prediction, label):
    #     #[B,1]
    #     loss=torch.
    #     for 


    def training_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx) #[B,L,N]
        # print(label[:,-1,-1])
        loss = self.loss_function(prediction[:,-1,-1],label[:,-1,-1]) #[B]
        
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)

        self.train_preds=torch.concat([self.train_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.train_labels=torch.concat([self.train_labels,label[:,-1,-1].cpu()],dim=0)

        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # print(f"梯度范数：{total_norm:.2f}")  # 正常范围：0.1-10.0
        # nn.functional.kl_div(pg_l, prob, reduction='batchmean')
        # print(pg_l,prob)
        # print(loss.shape)
        return loss
    
    
    # Vaild
    def on_validation_epoch_start(self):
        self.vaild_preds=torch.tensor([])
        self.vaild_labels=torch.tensor([])
        

    def on_validation_epoch_end(self):
        return None


    def validation_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx)
        
        # loss = self.loss_function(prediction[:,-1,-1],label[:,-1,-1])
        
        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)

        self.vaild_preds=torch.concat([self.vaild_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.vaild_labels=torch.concat([self.vaild_labels,label[:,-1,-1].cpu()],dim=0)


        
        loss=self.loss_function(self.vaild_preds,self.vaild_labels)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss




    #Test
    def on_test_start(self):
        self.test_preds=torch.tensor([])
        self.test_labels=torch.tensor([])
    
    def on_test_end(self):
        save_dir = self.hparams.test_result_path

        data=torch.concat([self.test_preds.reshape(-1,1),self.test_labels.reshape(-1,1)],dim=1)
        print(data.shape)


        #保存测试集结果
        df=pd.DataFrame(data,columns=["preds","labels"])
        df.to_csv(save_dir,index=False,encoding="UTF-8")



        #指标获取
        metric_save_path=self.hparams.metric_save_path
        dict_of_metric={
            "conf_hash":self.hparams.conf_hash,
        }
        header=["mode","NSE","MAE","KGE","RMSE","R"]
        metric_list=[]
        modes=["train","vaild","test"]
        for mode in modes:
            pred=None
            label=None
            if mode=="train":
                pred=self.train_preds
                label=self.train_labels
            elif mode=="vaild":
                pred=self.vaild_preds
                label=self.vaild_labels
            elif mode=="test":
                pred=self.test_preds
                label=self.test_labels
        
            _nse="{:.3f}".format(met.NSE(pred,label).item())
            _mae="{:.0f}".format(met.MAE(pred,label).item())
            _kge="{:.3f}".format(met.KGE(pred,label).item())
            _rmse="{:.0f}".format(met.RMSE(pred,label).item())
            _r="{:.3f}".format(met.R(pred,label).item())

            line=[mode,_nse,_mae,_kge,_rmse,_r]
            metric_list.append(line)
            print(line)

            dict_of_metric[mode+"_NSE"]=_nse
            dict_of_metric[mode+"_MAE"]=_mae
            dict_of_metric[mode+"_KGE"]=_kge
            dict_of_metric[mode+"_rmse"]=_rmse
            dict_of_metric[mode+"_r"]=_r

            df=pd.DataFrame(metric_list,columns=header)
            df.to_csv(metric_save_path,index=False,encoding="UTF-8")

            if mode=="test":
                self.update_metric_in_log(dict_of_metric)

        # np.savez(save_dir,preds=self.test_preds,labels=self.test_labels)


    def test_step(self, batch, batch_idx):
        prediction, label=self.forward(batch, batch_idx) #[B,L,N]
        
        
        prediction=self.inverse_transform_var(prediction)
        label=self.inverse_transform_var(label)


        self.test_preds=torch.concat([self.test_preds,prediction[:,-1,-1].cpu()],dim=0)
        self.test_labels=torch.concat([self.test_labels,label[:,-1,-1].cpu()],dim=0)
        # loss=self.loss_function(self.test_preds,self.test_labels)

        # self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        


        
    def update_metric_in_log(self,metric_dict):
        log_path=self.hparams.log_metric_csv_file

        if not os.path.exists(log_path):
            # 创建一个新的DataFrame并写入文件
            df = pd.DataFrame([metric_dict])
            df.to_csv(log_path, index=False)
        else:
            # 如果文件存在，读取文件并更新或新增数据
            df = pd.read_csv(log_path)
            if metric_dict["conf_hash"] in df['conf_hash'].values:
                # 更新指定ID对应的行
                df.loc[df['conf_hash'] == metric_dict["conf_hash"], metric_dict.keys()] = metric_dict.values()
            else:
                # 如果ID不存在，新增一行数据
                df = pd.concat([df, pd.DataFrame([metric_dict])], ignore_index=True)
            # 保存回CSV文件
            df.to_csv(log_path, index=False)



    def configure_loss(self):
        class RMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                
            def forward(self,yhat,y):
                return torch.sqrt(self.mse(yhat,y))

        class DplLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                
            def forward(self,yhat,y):
                loss=0.
                # print(yhat,y)
                for i in range(yhat.shape[0]):
                    if y[i].item() > 5.:
                        loss+=torch.exp(yhat[i:i+1]-y[i:i+1])-1
                    else:
                        loss+=torch.pow(yhat[i:i+1]-y[i:i+1],2)

                # print(torch.concat(loss,dim=0).shape)
                return loss/yhat.shape[0]


        self.loss_function = RMSELoss()


    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.lr, 
                betas=self.hparams.optimizer_betas, 
                weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(
                self.parameters(), 
                lr=self.hparams.lr, 
                max_iter=self.hparams.lr_max_iter)
        else:
            raise ValueError('Invalid optimizer type!')


        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, 
                    step_size=self.hparams.lr_step_size, 
                    gamma=self.hparams.lr_gamma)
            }
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            lr_scheduler = {
                "scheduler": lrs.MultiStepLR(
                    optimizer, 
                    milestones=self.hparams.milestones, 
                    gamma=self.hparams.gamma)
            }
        elif self.hparams.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=self.hparams.lrs_factor, 
                    patience=self.hparams.lrs_patience),
                "monitor": self.hparams.val_metric
            }
        elif self.hparams.lr_scheduler == 'WSD':
            assert self.hparams.lr_warmup_end_epochs < self.hparams.lr_stable_end_epochs < self.hparams.max_epochs

            def wsd_lr_lambda(epoch):
                if epoch < self.hparams.lr_warmup_end_epochs:
                    return (epoch + 1) / self.hparams.lr_warmup_end_epochs
                if self.hparams.lr_warmup_end_epochs <= epoch < self.hparams.lr_stable_end_epochs:
                    return 1.0
                if self.hparams.lr_stable_end_epochs <= epoch <= self.hparams.max_epochs:
                    return (epoch + 1 - self.hparams.lr_stable_end_epochs) / (
                            self.hparams.max_epochs - self.hparams.lr_stable_end_epochs)

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda),
            }
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }



    def load_model(self):
        model_name = self.hparams.model_name
        # Model = getattr(importlib.import_module('.' + model_name, package='core.model'), model_name)
        Model = getattr(importlib.import_module('.' + model_name, package='core.model'), model_name)
        self.model = self.instancialize(Model)
        

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        model_class_args = inspect.getfullargspec(Model.__init__).args[1:]  # 获取模型参数
        interface_args = self.hparams.keys()
        model_args_instance = {}
        for arg in model_class_args:
            if arg in interface_args:
                model_args_instance[arg] = getattr(self.hparams, arg)
        return Model(**model_args_instance)
    

    # MIN MAX
    def inverse_transform_var(self, data):
        if self.hparams.norm=="minmax":
            return self.min+(data+1.0)*(self.max-self.min)/2.0
        else: #"norm"
            return (data * self.std) + self.mean
    
    def transform_var(self, data):  
        if self.hparams.norm=="minmax":
            return -1.0+2*(data-self.min)/(self.max-self.min)
        else:
            return (data - self.mean) / self.std
    

    









