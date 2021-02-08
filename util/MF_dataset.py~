# By Yuxiang Sun, Dec. 13, 2020
# Email: sun.yuxiang@outlook.com

import os, torch, sys
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=240, input_w=320 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image  = np.array(PIL.Image.open(file_path)) # (w,h,c)
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        rgb = PIL.Image.fromarray(image[:,:,:3])
        thermal = image[:,:,-1]
        #thermal = thermal[..., np.newaxis]
            

        for t in self.transform:
            rgb = t(rgb)

        rgb = np.asarray(rgb.resize((self.input_w, self.input_h)), dtype=np.float32)[..., np.newaxis].transpose((2,0,1))/255
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32)[..., np.newaxis].transpose((2,0,1))/255

        # make rgb 3d grayscale
        rgb = np.repeat(rgb, 3, axis=0)
        
        return torch.tensor(rgb), torch.tensor(thermal), name

    def __len__(self):
        return self.n_data
