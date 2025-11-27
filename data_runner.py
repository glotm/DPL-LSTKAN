import os
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class HanJiangDataset(Dataset):

    def __init__(self,data_root,dataset_name,hist_len,pred_len,data_split,mode):
        """
        General TSF Dataset.

        Args:
            data_root(str): 1
            dataset_name(str): 2
            hist_len(int):
            pred_len(int):
            data_split(list):
            mode(str):
        """
        
        self.data_dir = os.path.join(data_root, dataset_name)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.train_len, self.val_len, self.test_len = data_split
        self.preditct_len=self.train_len+self.val_len+self.test_len
        # self.freq = freq
        self.mode=mode

        assert(mode in ["train","valid","test","predict"],"mode {} mismatch, should be in [train, valid, test,predict]".format(mode))
        mode_map = {'train': 0, 'valid': 1, 'test': 2, "predict":3}
        self.set_type = mode_map[mode]


        self.var, self.time_marker = self.__read_data__()

    
    def __read_data__(self):
        feature_path = os.path.join(self.data_dir, 'feature.npz')
        feature = np.load(feature_path)

        var = feature['var']
        time_marker = feature['time_marker']


        if self.set_type==3:
            var = var[0:self.preditct_len]
            time_marker = time_marker[0:self.preditct_len]
        else:
            border1s = [0, self.train_len, self.train_len + self.val_len]
            border2s = [self.train_len, self.train_len + self.val_len, self.train_len + self.val_len + self.test_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            var = var[border1:border2]
            time_marker = time_marker[border1:border2]

 
             

        return var, time_marker
    


    def __getitem__(self, index):
        hist_start = index
        hist_end = index + self.hist_len
        pred_end = hist_end + self.pred_len

        var_x = self.var[hist_start:hist_end, ...]
        marker_x = self.time_marker[hist_start:hist_end, ...]

        var_y = self.var[hist_end:pred_end, ...]
        marker_y = self.time_marker[hist_end:pred_end, ...]
        return var_x, marker_x, var_y, marker_y

    def __len__(self):
        return len(self.var) - (self.hist_len + self.pred_len) + 1
    


def data_provider(config, mode):
    return HanJiangDataset(
        data_root=config['data_root'],
        dataset_name=config['dataset_name'],
        hist_len=config['hist_len'],
        pred_len=config['pred_len'],
        data_split=config['data_split'],
        # freq=config['freq'],
        mode=mode,
    )



class DataInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs

    def train_dataloader(self):
        train_set = data_provider(self.kwargs, mode='train')
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        val_set = data_provider(self.kwargs, mode='valid')
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        test_set = data_provider(self.kwargs, mode='test')
        return DataLoader(test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)
    
