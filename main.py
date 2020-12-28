import sys
import os
import shutil
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter
from sklearn.model_selection import train_test_split

from dataset import MNIST3dDataset, get_dataloaders
from model import MNIST3dModel

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MNIST3dClassifier(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.train_loss_metric = AverageMeter()
        self.train_acc_metric = AverageMeter()
        self.val_loss_metric = AverageMeter()
        self.val_acc_metric = AverageMeter()


    def setup_data(self, path):
        with h5py.File(path, 'r') as hf:
            X_train = hf['X_train'][:]
            y_train = hf['y_train'][:]
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15,
                                                          random_state=1, shuffle=True)
        train_loader = get_dataloaders(X_data=X_train, y_data=y_train,
                                       batch_size=self.config['batch_size'],
                                       num_workers=self.config['dataset']['num_workers'])
        val_loader = get_dataloaders(X_data=X_val, y_data=y_val,
                                     batch_size=self.config['batch_size'],
                                     num_workers=self.config['dataset']['num_workers'])
        return train_loader, val_loader


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device
    
    
    def _step(self, model, criterion, data, targets):
        output = model(data)
        targets = targets.view(-1)
        
        loss = criterion(output, targets.type(torch.cuda.LongTensor))

        _, predicted = torch.max(output.data, 1)

        acc = torch.mean((predicted == targets).type(torch.FloatTensor))
        
        return loss, acc
    
    
    def train(self):
        train_loader, val_loader = self.setup_data(self.config['dataset']['path'])

        model = MNIST3dModel(num_classes=self.config['dataset']['num_classes'])
        model.to(self.device)
        model = self._load_pre_trained_weights(model)

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(1, self.config['epochs'] + 1):
            for data, targets in train_loader:
                optimizer.zero_grad()

                data = data.to(self.device)
                targets = targets.to(self.device)

                loss, acc = self._step(model, criterion, data, targets)

                self.train_loss_metric.update(loss.item())
                self.train_acc_metric.update(acc.item())
                
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss',
                                           self.train_loss_metric.avg,
                                           global_step=n_iter)
                    self.writer.add_scalar('train_acc',
                                           self.train_acc_metric.avg,
                                           global_step=n_iter)
                    print('[{}/{}] loss: {:.2f}, acc {:.2f}'.format(epoch_counter,
                                                                self.config['epochs'],
                                                                self.train_loss_metric.avg,
                                                                self.train_acc_metric.avg))

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                self.val_loss_metric.reset()
                self.val_acc_metric.reset()
                
                valid_loss, valid_acc = self._validate(model, criterion, val_loader)
                
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss',
                                       self.val_loss_metric.avg,
                                       global_step=valid_n_iter)
                self.writer.add_scalar('validation_acc',
                                       self.val_acc_metric.avg,
                                       global_step=valid_n_iter)
                print('[{}/{}] val_loss: {:.2f}, val_acc {:.2f}'.format(epoch_counter,
                                                                self.config['epochs'],
                                                                self.val_loss_metric.avg,
                                                                self.val_acc_metric.avg))
                valid_n_iter += 1

            scheduler.step(valid_loss)
            self.writer.add_scalar('lr_reduce_on_plateau',
                                   [group['lr'] for group in optimizer.param_groups][0],
                                   global_step=n_iter)


    def _validate(self, model, criterion, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            valid_acc = 0.0
            counter = 0
            for data, targets in valid_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                loss, acc = self._step(model, criterion, data, targets)
                valid_loss += loss.item()
                valid_acc += acc.item()
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(acc.item())
                counter += 1
            valid_loss /= counter
            valid_acc /= counter
        model.train()
        return valid_loss, valid_acc


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


    def setup_test(self, path):
        with h5py.File(path, 'r') as hf:
            X_test = hf['X_test'][:]
            y_test = hf['y_test'][:]
        test_loader = get_dataloaders(X_data=X_test, y_data=y_test)
        return test_loader

    def test(self):
        test_loader = self.setup_test(self.config['dataset']['path'])
        
        model = MNIST3dModel(num_classes=self.config['dataset']['num_classes'])
        model.to(self.device)
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        state_dict = torch.load(os.path.join(model_checkpoints_folder, 'model.pth'))
        model.load_state_dict(state_dict)

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            model.eval()

            test_acc = 0.0
            counter = 0
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                _, acc = self._step(model, criterion, data, targets)
                test_acc += acc.item()
                counter += 1
            test_acc /= counter

        print(f'\n\nTest accuracy: {test_acc:.2f}')
