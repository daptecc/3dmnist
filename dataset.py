import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib.pyplot import cm
import numpy as np

class MNIST3dDataset(Dataset):
    '''
    '''
    def __init__(self, X_data, y_data):
        '''
        add RGB dimension and reshape to 1 + 4D
        convert labels to ohe
        '''
        self.X_data = self.add_rgb_to_data(X_data)
        self.y_data = y_data
    
    def __getitem__(self, idx):

        X_data = torch.tensor(self.X_data[idx])
        X_data = X_data.reshape(16,16,16,3)
        X_data = X_data.permute(3,0,1,2)
        X_data = X_data.type(torch.FloatTensor)
        y_data = torch.tensor(np.float32(self.y_data[idx]))
        return X_data, y_data
               
    def __len__(self):
        return self.X_data.shape[0]
    
    def add_rgb_dimension(self, array):
        '''
        translate data to color
        '''
        scaler_map = cm.ScalarMappable(cmap="Oranges")
        array = scaler_map.to_rgba(array)[:, : -1]
        return array
    
    def add_rgb_to_data(self, data):
        '''
        iterate dataset, add rgb dimension
        '''
        data_w_rgb = np.ndarray((data.shape[0], data.shape[1], 3))
        for i in range(data.shape[0]):
            data_w_rgb[i] = self.add_rgb_dimension(data[i])
        return data_w_rgb


def get_dataloaders(X_data, y_data, batch_size=64, num_workers=4):
    '''
    A handy function to make our dataloaders
    '''
    dataset = MNIST3dDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader