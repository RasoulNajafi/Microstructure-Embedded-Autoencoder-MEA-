import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
from tqdm import  tqdm
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages

bath_path = r'C:\temp\UNet_SH'
morph_path_HR = r'C:\temp\UNet_SH\dataset\morph_dataSet_HR\morph_dataSet_HR.txt'  
FE_path_HR = r'C:\temp\UNet_SH\dataset\output_FE_HR\output_FE_HR.txt'  
FE_path_LR = r'C:\temp\UNet_SH\dataset\output_FE_LR\output_FE_LR.txt'
morph_path_LR = r'C:\temp\UNet_SH\dataset\morph_dataSet_LR\morph_dataSet_LR.txt' 
morph_path_51R = r"C:\temp\UNet_SH\dataset\morph_dataSet_51R\morph_dataSet_51x51.txt"
morph_path_26R = r"C:\temp\UNet_SH\dataset\morph_dataSet_26R\morph_dataSet_26x26.txt"
morph_path_13R = r"C:\temp\UNet_SH\dataset\morph_dataSet_13R\morph_dataSet_13x13.txt"

morph_HR = np.loadtxt(morph_path_HR) 
FE_HR = np.loadtxt(FE_path_HR)   
morph_LR = np.loadtxt(morph_path_LR) 
FE_LR = np.loadtxt(FE_path_LR)
morph_51R = np.loadtxt(morph_path_51R)
morph_26R = np.loadtxt(morph_path_26R)
morph_13R = np.loadtxt(morph_path_13R)    

###############################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True   
    np.random.seed(seed)   
    random.seed(seed)
set_seed(1101)

def split_data(Data, resolution):
    lst = list()
    s_num  = Data.shape[0]
    for i in range(s_num):
        X = Data[i, :] 
        X = X.reshape(resolution , resolution)
        X = np.expand_dims(X, axis=0)
        lst.append(X)
    return lst

morph_HR = split_data(morph_HR, 101)
FE_HR = split_data(FE_HR, 101)
morph_LR = split_data(morph_LR, 11)
FE_LR = split_data(FE_LR, 11)
morph_51R = split_data(morph_51R, 51)
morph_26R = split_data(morph_26R, 26)
morph_13R = split_data(morph_13R, 13)

def shuffle_dataset_training(morph_data_HR, FE_data_HR, morph_data_LR, FE_data_LR, morph_51R, morph_26R, morph_13R):     
    morph_data_HR = np.vstack(morph_data_HR)
    FE_data_HR = np.vstack(FE_data_HR)    
    morph_data_LR = np.vstack(morph_data_LR)
    FE_data_LR = np.vstack(FE_data_LR)   
    morph_51R = np.vstack(morph_51R)  
    morph_26R = np.vstack(morph_26R)  
    morph_13R = np.vstack(morph_13R)  
    # Shuffle data
    state = np.random.get_state()
    np.random.shuffle(morph_data_HR)
    np.random.set_state(state)
    np.random.shuffle(FE_data_HR)
    np.random.set_state(state)
    np.random.shuffle(morph_data_LR)
    np.random.set_state(state)
    np.random.shuffle(FE_data_LR)
    np.random.set_state(state)
    np.random.shuffle(morph_51R)
    np.random.set_state(state)
    np.random.shuffle(morph_26R)
    np.random.set_state(state)
    np.random.shuffle(morph_13R)
    
    save_Path = r'C:\temp\UNet_SH\train' 
    np.save(save_Path + r'\morph_data_HR', morph_data_HR)
    np.save(save_Path+ r'\FE_data_HR', FE_data_HR)
    np.save(save_Path + r'\morph_data_LR', morph_data_LR)
    np.save(save_Path+ r'\FE_data_LR', FE_data_LR)
    np.save(save_Path+ r'\morph_51R', morph_51R)
    np.save(save_Path+ r'\morph_26R', morph_26R)
    np.save(save_Path+ r'\morph_13R', morph_13R)
       
shuffle_dataset_training(morph_HR, FE_HR, morph_LR, FE_LR, morph_51R, morph_26R, morph_13R)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder layers
        # Initial Image:  A×3 (Width x Height x Channels)
        # Input image size: 1 x 11 x 11
        self.enc_conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU()
                                       )
        # Input image size: 16 x 11 x 11
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm2d(16),
                                               nn.ReLU()
                                       )
        # Input image size 16 x 11 x 11        
        self.enc_conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU()
                                       )
        # Input image size 32 x 11 x 11 
        self.enc_conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU()
                                       )
        # Input image size 32 x 11 x 11 
        self.enc_conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )
        # Input image size 64 x 11 x 11 
        self.enc_conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )
        # Input image size 64 x 11 x 11 
        self.enc_conv7 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU()
                                       )


        # Decoder layers
        #Input image size 128 x 11 x 11
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        #Input image size 128 x 11 x 11
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #Input image size 128 x 13 x 13 +1
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(129, 129, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(129),
            nn.ReLU()
        )
                
        self.upsample1 = nn.ConvTranspose2d(129, 129, kernel_size=2, stride=2, padding=0)
            
        #Input image size (129 x 26 x 26) + 1
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(130, 130, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(130),
            nn.ReLU()
        )
        
        #Input image size (130 x 26 x 26)
        self.dec_conv5 = nn.Sequential(                  #####################
            nn.Conv2d(130, 130, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(130),
            nn.ReLU()
        )
        #Input image size (130 x 26 x 26)
        self.dec_conv6 = nn.Sequential(                  #####################
            nn.Conv2d(130, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #Input image size (64 x 26 x 26)
        self.dec_conv7 = nn.Sequential(                  #####################
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )        
        #Input image size (64 x 26 x 26)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        
        #Input image size (64 x 51 x 51)
        
        self.dec_conv8 = nn.Sequential(
            nn.Conv2d(65, 65, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(65),
            nn.ReLU()
        )
        self.dec_conv9 = nn.Sequential(
            nn.Conv2d(65, 65, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(65),
            nn.ReLU()
        )        
        #Input image size (32 x 51 x 51)
        self.dec_conv10 = nn.Sequential(                  
            nn.Conv2d(65, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec_conv11 = nn.Sequential(                   
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )        
        
        #Input image size (32 x 51 x 51)
        self.upsample3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        
        #Input image size (32 x 101 x 101)  + (1,101,101)
        self.dec_conv12 = nn.Sequential(
            nn.Conv2d(33, 33, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(33),
            nn.ReLU()
        )
        #Input image size (33 x 101 x 101)
        self.dec_conv13 = nn.Sequential(
            nn.Conv2d(33, 33, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(33),
            nn.ReLU()
        )
        #Input image size (33 x 101 x 101)
        self.dec_conv14 = nn.Sequential(            
            nn.Conv2d(33, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        #Input image size (16 x 101 x 101)
        self.dec_conv15 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) 
        self.dec_final = nn.Conv2d(8, 1, kernel_size=1) 

    def forward(self, x, morph_HR, morph_13R, morph_26R, morph_51R):
        # Encoder
        enc1 = self.enc_conv1(x)
        #print("After enc_conv1:", enc1.size())        
        enc2 = self.enc_conv2(enc1)
        #print("After enc_conv2:", enc2.size())        
        enc3 = self.enc_conv3(enc2)
        #print("After enc_conv3:", enc3.size())        
        enc4 = self.enc_conv4(enc3)
        #print("After enc_conv4:", enc4.size())        
        enc5 = self.enc_conv5(enc4)
        #print("After enc_conv5:", enc5.size())        
        enc6 = self.enc_conv6(enc5)
        #print("After enc_conv6:", enc6.size())        
        out = self.enc_conv7(enc6)
        #print("After enc_conv7:", enc7.size())        
        
        # Decoder
        dec1 = self.dec_conv1(out)
        dec2 = self.dec_conv2(dec1)
        
        conct_layer1 = torch.cat((dec2, morph_13R), 1) 
        
        dec3 = self.dec_conv3(conct_layer1)        
        upsampled_1 = self.upsample1(dec3)
        
        conct_layer2 = torch.cat((upsampled_1, morph_26R), 1)
        
        dec4 = self.dec_conv4(conct_layer2)
        dec5 = self.dec_conv5(dec4)
        dec6 = self.dec_conv6(dec5)
        dec7 = self.dec_conv7(dec6)
        
        upsampled_2 = self.upsample2(dec7)
        upsampled_2 = F.adaptive_avg_pool2d(upsampled_2, (51, 51)) 
        conct_layer3 = torch.cat((upsampled_2, morph_51R), 1)
        
        dec8 = self.dec_conv8(conct_layer3)
        dec9 = self.dec_conv9(dec8)
        dec10 = self.dec_conv10(dec9)
        dec11 = self.dec_conv11(dec10)

        upsampled_3 = self.upsample3(dec11)
        upsampled_3 = F.adaptive_avg_pool2d(upsampled_3, (101, 101))         
        conct_layer4 = torch.cat((upsampled_3, morph_HR), 1)  
        
        dec12 = self.dec_conv12(conct_layer4)
        dec13 = self.dec_conv13(dec12)
        dec14 = self.dec_conv14(dec13)
        dec15 = self.dec_conv15(dec14)
        out_final = self.dec_final(dec15)

        return out_final


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.morphology_HRbox = list()
        self.morphology_13Rbox = list()
        self.morphology_26Rbox = list()
        self.morphology_51Rbox = list()
        self.FE_HRbox = list()
        self.FE_LRbox = list()       
        
        morphology_HR =  np.load(r'C:\temp\UNet_SH\train\morph_data_HR.npy' )
        FE_data_HR =  np.load(r'C:\temp\UNet_SH\train\FE_data_HR.npy' )
        FE_data_LR =  np.load(r'C:\temp\UNet_SH\train\FE_data_LR.npy' )
        morphology_13R =  np.load(r'C:\temp\UNet_SH\train\morph_13R.npy' )
        morphology_26R =  np.load(r'C:\temp\UNet_SH\train\morph_26R.npy' )
        morphology_51R =  np.load(r'C:\temp\UNet_SH\train\morph_51R.npy' )
        
        train_num = int(0.9 * morphology_HR.shape[0])
        val_num = int(0.1 * morphology_HR.shape[0])
               
        if self.mode=='Train':
            morphology_HR  = morphology_HR[0:train_num]
            FE_data_HR = FE_data_HR[0:train_num]
            FE_data_LR = FE_data_LR[0:train_num]
            morphology_13R  = morphology_13R[0:train_num]
            morphology_26R  = morphology_26R[0:train_num]
            morphology_51R  = morphology_51R[0:train_num]
            
        if self.mode == 'Valid':
            morphology_HR  = morphology_HR[ train_num : train_num + val_num]
            FE_data_HR = FE_data_HR[ train_num : train_num + val_num]
            FE_data_LR = FE_data_LR[ train_num : train_num + val_num]
            morphology_13R  = morphology_13R[ train_num : train_num + val_num]
            morphology_26R  = morphology_26R[ train_num : train_num + val_num]
            morphology_51R  = morphology_51R[ train_num : train_num + val_num]
            
        for line in range(morphology_HR.shape[0]):
            self.morphology_HRbox.append(morphology_HR[line])
            self.morphology_13Rbox.append(morphology_13R[line])
            self.morphology_26Rbox.append(morphology_26R[line])
            self.morphology_51Rbox.append(morphology_51R[line])
            self.FE_HRbox.append(FE_data_HR[line])
            self.FE_LRbox.append(FE_data_LR[line])           
            
    def __len__(self):
        return len(self.morphology_HRbox)

    def __getitem__(self, index):
        morph_HR = self.morphology_HRbox[index]
        morph_HR = np.array(morph_HR, dtype='float')
        morph_HR = torch.from_numpy(morph_HR)
        morph_HR = morph_HR.type(torch.FloatTensor)
        
        morph_13R = self.morphology_13Rbox[index]
        morph_13R = np.array(morph_13R, dtype='float')
        morph_13R = torch.from_numpy(morph_13R)
        morph_13R = morph_13R.type(torch.FloatTensor)
        
        morph_26R = self.morphology_26Rbox[index]
        morph_26R = np.array(morph_26R, dtype='float')
        morph_26R = torch.from_numpy(morph_26R)
        morph_26R = morph_26R.type(torch.FloatTensor)
        
        morph_51R = self.morphology_51Rbox[index]
        morph_51R = np.array(morph_51R, dtype='float')
        morph_51R = torch.from_numpy(morph_51R)
        morph_51R = morph_51R.type(torch.FloatTensor)
        
        FE_HR = self.FE_HRbox[index]
        FE_HR = np.array(FE_HR, dtype='float')
        FE_HR = torch.from_numpy(FE_HR)
        FE_HR = FE_HR.type(torch.FloatTensor)
        
        FE_LR = self.FE_LRbox[index]
        FE_LR = np.array(FE_LR, dtype='float')
        FE_LR = torch.from_numpy(FE_LR)
        FE_LR = FE_LR.type(torch.FloatTensor)
        return morph_HR, morph_13R, morph_26R, morph_51R, FE_LR, FE_HR


mode='Train'
batch_size_train = 200  
batch_size_valid = 200
 

if mode=='Train':
    train_dataset = MyDataset('Train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
    
    validate_dataset = MyDataset('Valid')
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size_valid)

os.chdir(r'K:\Najafi\codes\UNet_SH')
MSELoss = torch.nn.MSELoss()
n_epochs = 500
net = UNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-04)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
if torch.cuda.is_available():
    print('GPU is available')
    net = net.cuda()
else:
    print('GPU is not available')
        
loss_plot = []
loss_validate_plot = []
if __name__ == '__main__':
    if mode =='Train':
        model_name = str(time.localtime(time.time())[2])+'-'+str(time.localtime(time.time())[1])+'-'+str(time.localtime(time.time())[0])
        dir_name = 'model_save' + '/' + model_name
        try:
            os.makedirs(dir_name)
        except OSError:
            pass

        try:
            os.makedirs(r'fig_save/save_process/')
        except OSError:
            pass
        with open(dir_name + '/' + 'data.txt', 'a+') as f:
            for epoch in range(1, n_epochs + 1):
                net.train()
                for step, (morph_Hbatch, morph_13batch, morph_26batch , morph_51batch,  FE_LR_batch,  FE_HR_batch) in enumerate(tqdm(train_loader, desc=f"Training_epoch{epoch}")):
                    if torch.cuda.is_available():
                        morph_Hbatch = torch.unsqueeze(morph_Hbatch, 1)
                        morph_13batch = torch.unsqueeze(morph_13batch, 1)
                        morph_26batch = torch.unsqueeze(morph_26batch, 1)
                        morph_51batch = torch.unsqueeze(morph_51batch, 1)
                        FE_LR_batch = torch.unsqueeze(FE_LR_batch, 1)
                        FE_HR_batch = torch.unsqueeze(FE_HR_batch, 1)
                        morph_Hbatch, morph_13batch, morph_26batch, morph_51batch, FE_LR_batch, FE_HR_batch = morph_Hbatch.cuda(), morph_13batch.cuda(), morph_26batch.cuda(), morph_51batch.cuda(), FE_LR_batch.cuda(), FE_HR_batch.cuda()
                    optimizer.zero_grad()
                    prediction_origin = net(FE_LR_batch, morph_Hbatch, morph_13batch, morph_26batch, morph_51batch)
                    loss = MSELoss(FE_HR_batch, prediction_origin)
                    loss.backward()
                    optimizer.step()
        
                net.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for step_valid, (morph_Hvalid, morph_13valid , morph_26valid,  morph_51valid, FE_LR_valid, FE_HR_valid) in enumerate(validate_loader):
                        if torch.cuda.is_available():
                            morph_Hvalid = torch.unsqueeze(morph_Hvalid, 1)
                            morph_13valid = torch.unsqueeze(morph_13valid, 1)
                            morph_26valid = torch.unsqueeze(morph_26valid, 1)
                            morph_51valid = torch.unsqueeze(morph_51valid, 1)
                            FE_LR_valid = torch.unsqueeze(FE_LR_valid, 1)
                            FE_HR_valid = torch.unsqueeze(FE_HR_valid, 1)
                            
                            morph_Hvalid, morph_13valid, morph_26valid, morph_51valid, FE_LR_valid, FE_HR_valid = morph_Hvalid.cuda(), morph_13valid.cuda(), morph_26valid.cuda(), morph_51valid.cuda(), FE_LR_valid.cuda(), FE_HR_valid.cuda()      
                        prediction_validate = net(FE_LR_valid, morph_Hvalid, morph_13valid, morph_26valid, morph_51valid)
                        loss_validate = MSELoss(FE_HR_valid, prediction_validate)
                        total_val_loss += loss_validate.item()
                    plt.figure(1, figsize=(10, 10))
                    plt.subplot(2, 2, 1)
                    plt.imshow(FE_HR_batch[0, 0, :, :].cpu().data.numpy(), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
                    plt.title('GT:Training')
                    plt.colorbar()
                    plt.subplot(2, 2, 2)
                    plt.imshow(prediction_origin[0, 0, :, :].cpu().data.numpy(), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
                    plt.title('UNet: Training')
                    plt.colorbar()
                    plt.subplot(2, 2, 3)
                    plt.imshow(FE_HR_valid[0, 0, :, :].cpu().data.numpy(), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
                    plt.title('GT: Validation')
                    plt.colorbar()
                    plt.subplot(2, 2, 4)
                    plt.imshow(prediction_validate[0, 0, :, :].cpu().data.numpy(), cmap='viridis', extent=[0, 1, 0, 1], origin='lower', aspect='auto')
                    plt.title('UNet: Validation')
                    plt.colorbar()

                    plt.savefig(f'fig_save/save_process/{epoch}.png')
                    plt.close('all')    
                    f.write(f"iter_num: {epoch}      loss: {loss.item():.8f}    loss_validate: {loss_validate.item():.8f}  \r\n")
                    torch.save(net.state_dict(), f'{dir_name}/net_{epoch}_epoch.pkl')
                    torch.save(optimizer.state_dict(), f'{dir_name}/optimizer_{epoch}_epoch.pkl')
                    loss_plot.append(loss.item())
                    loss_validate_plot.append(loss_validate.item())
                
                average_val_loss = total_val_loss / len(validate_loader)
                #scheduler.step(average_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}/{n_epochs}, Training Loss: {loss.item():.8f}, Validation Loss: {average_val_loss:.8f}, Learning Rate: {current_lr}")
        
            best_epoch = (loss_validate_plot.index(min(loss_validate_plot))) 
            print(f"The ANN has been trained, the best epoch is {best_epoch}")
            
################################################################################
morph_HR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\test_morph_HR.txt") 
morph_13R = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\test_morph_13x13.txt")
morph_26R = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\test_morph_26x26.txt")
morph_51R = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\test_morph_51x51.txt")
FE_HR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\output_FE_test_HR.txt") 
FE_LR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\output_FE_test_LR.txt")

morph_HR = split_data(morph_HR,101)
morph_13R = split_data(morph_13R,13)
morph_26R = split_data(morph_26R,26)
morph_51R = split_data(morph_51R,51)
FE_HR = split_data(FE_HR,101)
FE_LR = split_data(FE_LR,11)

net.load_state_dict(torch.load(r"K:\Najafi\codes\UNet_SH\UNet_V3_type1\net_477_epoch.pkl"))
optimizer.load_state_dict(torch.load(r"K:\Najafi\codes\UNet_SH\UNet_V3_type1\optimizer_477_epoch.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)   
net.eval()

with PdfPages('FluxUNet_type1.pdf') as pdf:
    for i in range(len(morph_HR)):
        morphology = morph_HR[i]
        morphology = np.expand_dims(morphology, axis=0)
        morphology = torch.from_numpy(morphology.astype(np.float32))
        morphology = morphology.type(torch.FloatTensor)
        morphology = morphology.to(device) 
        
        morphology13 = morph_13R[i]
        morphology13 = np.expand_dims(morphology13, axis=0)
        morphology13 = torch.from_numpy(morphology13.astype(np.float32))
        morphology13 = morphology13.type(torch.FloatTensor)
        morphology13 = morphology13.to(device)
        
        morphology26 = morph_26R[i]    
        morphology26 = np.expand_dims(morphology26, axis=0)
        morphology26 = torch.from_numpy(morphology26.astype(np.float32))
        morphology26 = morphology26.type(torch.FloatTensor)
        morphology26 = morphology26.to(device)
        
        morphology51 = morph_51R[i]    
        morphology51 = np.expand_dims(morphology51, axis=0)
        morphology51 = torch.from_numpy(morphology51.astype(np.float32))
        morphology51 = morphology51.type(torch.FloatTensor)
        morphology51 = morphology51.to(device)
        
        FE_L = FE_LR[i]
        FE_L = np.expand_dims(FE_L, axis=0)
        FE_L = torch.from_numpy(FE_L.astype(np.float32))
        FE_L = FE_L.type(torch.FloatTensor)
        FE_L = FE_L.to(device)
            
        
        FE_H = FE_HR[i]
        FE_H = torch.from_numpy(FE_H.astype(np.float32))
        FE_H = FE_H.to(device)       
        test = net(FE_L, morphology, morphology13, morphology26, morphology51)
        
        N_HR = 101
        Ne = N_HR-1
        L = 1.0  # Length of the square domain
        dx = L / Ne  # grid spacing
        
        # Create a meshgrid
        xx = np.linspace(0, L, N_HR)
        yy = np.linspace(0, L, N_HR)
        X, Y = np.meshgrid(xx, yy)
        
        conductivity_map = morphology.cpu().data.numpy()
        T_mesh1 = test.cpu().data.numpy()
        T_mesh2 = FE_H.cpu().data.numpy()
        
        dT_dx_HR1 = -conductivity_map[0][0] * np.gradient(T_mesh1[0][0], dx, axis=1)  # Heat flux in x-direction
        dT_dy_HR1 = -conductivity_map[0][0] * np.gradient(T_mesh1[0][0], dx, axis=0) 
        
        dT_dx_HR2 = -conductivity_map[0][0] * np.gradient(T_mesh2[0], dx, axis=1)  # Heat flux in x-direction
        dT_dy_HR2 = -conductivity_map[0][0] * np.gradient(T_mesh2[0], dx, axis=0)# Heat flux in y-direction
        
        # Calculate the flux magnitude
        flux_magnitude_HR1 = np.sqrt(dT_dx_HR1**2 + dT_dy_HR1**2)
        flux_magnitude_HR2 = np.sqrt(dT_dx_HR2**2 + dT_dy_HR2**2)
        
        fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))
        error_abs=np.abs(flux_magnitude_HR1 - flux_magnitude_HR2)
        error_mean = np.mean(error_abs)
        titles = ['Morphology', 'UNet: test', 'GT: FE', '(GT - UNet)']
        data = [
            morphology[0, 0, :, :].cpu().data.numpy(),
            flux_magnitude_HR1,
            flux_magnitude_HR2,
            np.abs(FE_H[0].cpu().data.numpy() - test[0, 0, :, :].cpu().data.numpy())
               
        ]
        colormaps = ['viridis', 'hot', 'hot', 'hot', 'hot']
        
        for ax, idx, d, cmap in zip(axs, range(len(axs)), data, colormaps):
            if idx in [3]: 
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=0, vmax=0.15)
            elif idx in [2]:  
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=0, vmax=4) 
            elif idx in [1]: 
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=0, vmax=4)  
            else:
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
            ax.set_title(title)
            ax.set_xticks([])   
            ax.set_yticks([])   
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30)  
        
        plt.tight_layout()
        pdf.savefig(fig)   
        plt.close(fig)   

os.chdir(r"K:\Najafi\codes\UNet_SH")
with PdfPages('UNet_V3_type1.pdf') as pdf:
    for i in range(len(morph_HR)):
        morphology = morph_HR[i]
        morphology = np.expand_dims(morphology, axis=0)
        morphology = torch.from_numpy(morphology.astype(np.float32))
        morphology = morphology.type(torch.FloatTensor)
        morphology = morphology.to(device) 
        
        morphology13 = morph_13R[i]
        morphology13 = np.expand_dims(morphology13, axis=0)
        morphology13 = torch.from_numpy(morphology13.astype(np.float32))
        morphology13 = morphology13.type(torch.FloatTensor)
        morphology13 = morphology13.to(device)
        
        morphology26 = morph_26R[i]    
        morphology26 = np.expand_dims(morphology26, axis=0)
        morphology26 = torch.from_numpy(morphology26.astype(np.float32))
        morphology26 = morphology26.type(torch.FloatTensor)
        morphology26 = morphology26.to(device)
        
        morphology51 = morph_51R[i]    
        morphology51 = np.expand_dims(morphology51, axis=0)
        morphology51 = torch.from_numpy(morphology51.astype(np.float32))
        morphology51 = morphology51.type(torch.FloatTensor)
        morphology51 = morphology51.to(device)
        
        FE_L = FE_LR[i]
        FE_L = np.expand_dims(FE_L, axis=0)
        FE_L = torch.from_numpy(FE_L.astype(np.float32))
        FE_L = FE_L.type(torch.FloatTensor)
        FE_L = FE_L.to(device)            
        
        FE_H = FE_HR[i]
        FE_H = torch.from_numpy(FE_H.astype(np.float32))
        FE_H = FE_H.to(device)       
        test = net(FE_L, morphology, morphology13, morphology26, morphology51)
        
        fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))   
        error = np.abs(FE_H[0].cpu().data.numpy() - test[0, 0, :, :].cpu().data.numpy())
        mean_error = np.mean(error)
        titles = ['Morphology', 'UNet: test', 'GT: FE', '(GT - UNet)']
        data = [
            morphology[0, 0, :, :].cpu().data.numpy(),
            test[0, 0, :, :].cpu().data.numpy(),
            FE_H[0].cpu().data.numpy(),
            np.abs(FE_H[0].cpu().data.numpy() - test[0, 0, :, :].cpu().data.numpy())
        ]
        colormaps = ['viridis', 'coolwarm', 'coolwarm', 'coolwarm']
        
        for ax, idx, d, cmap in zip(axs, range(len(axs)), data, colormaps):
            if idx in [3]:   
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=0, vmax=0.15)   
            else:
                cax = ax.imshow(d, cmap=cmap, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
            ax.set_title(title)
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks
            ax.set_title(title, fontsize=30)
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30)  #        
        plt.tight_layout()
        pdf.savefig(fig)   
        #plt.savefig(r"C:\temp\UNet_SH\UNet_V3_type2.png", dpi=300)
        plt.close(fig)  