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
import time

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
        # Input image size: 1 x 101 x 101
        self.enc_conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU()
                                       )
        # Input image size: 16 x 101 x 101        
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU()
                                       )
        # Input image size: 16 x 101 x 101
        self.enc_conv3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                                               nn.BatchNorm2d(16),
                                               nn.ReLU()
                                       )
        # Input image size 16 x 51 x 51        
        self.enc_conv4 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU()
                                       )
        self.enc_conv5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU()
                                       )
        # Input image size 32 x 51 x 51 
        self.enc_conv6 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU()
                                       )
        # Input image size 32 x 26 x 26 
        self.enc_conv7 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )
        # Input image size 64 x 26 x 26 
        self.enc_conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )
        # Input image size 64 x 26 x 26 
        self.enc_conv9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )
        # Input image size 64 x 13 x 13 
        self.enc_conv10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU()
                                       )
        # Input image size 128 x 13 x 13 
        self.enc_conv11 = nn.Sequential(nn.Conv2d(128 , 128, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU()
                                       )
        # Input image size 128 x 13 x 13 
        self.enc_conv12 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU()
                                       )
# Final output image size 128 x 11 x 11 


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
        #Input image size 128 x 13 x 13  + 128 von enc11
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        #Input image size 128 x 13 x 13
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
            
        #Input image size (128 x 26 x 26)
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )        
        #Input image size (64 x 26 x 26)
        self.dec_conv6 = nn.Sequential(                  #####################
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )        
        #Input image size (64 x 26 x 26)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        
        #Input image size (64 x 51 x 51)
        self.dec_conv7 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        #Input image size (32 x 51 x 51)
        self.dec_conv8 = nn.Sequential(                  #######################
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        #Input image size (32 x 51 x 51)
        self.upsample3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        
        #Input image size (32 x 101 x 101)  
        self.dec_conv9 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        #Input image size (16 x 101 x 101)
        self.dec_conv10 = nn.Sequential(           ########################
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        #Input image size (16 x 101 x 101)
        self.dec_conv11 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ) 
        self.dec_final = nn.Conv2d(4, 1, kernel_size=1)     

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        #print("After enc_conv1:", enc1.size())        
        enc2 = self.enc_conv2(enc1)     
        enc3 = self.enc_conv3(enc2)        
        enc4 = self.enc_conv4(enc3)       
        enc5 = self.enc_conv5(enc4)       
        enc6 = self.enc_conv6(enc5)        
        enc7 = self.enc_conv7(enc6)   
        enc8 = self.enc_conv8(enc7) 
        enc9 = self.enc_conv9(enc8) 
        enc10 = self.enc_conv10(enc9) 
        enc11 = self.enc_conv11(enc10)
        out = self.enc_conv12(enc11)
        
        
        # Decoder
        dec1 = self.dec_conv1(out)
        #print("After dec_conv1:", dec1.size())
        dec2 = self.dec_conv2(dec1) 
        
        conct_layer1 = torch.cat((dec2, enc11), 1)
        
        dec3 = self.dec_conv3(conct_layer1)
        dec4 = self.dec_conv4(dec3)                 
        upsampled_1 = self.upsample1(dec4)
        
        conct_layer2 = torch.cat((upsampled_1, enc8), 1)
        
        dec5 = self.dec_conv5(conct_layer2)
        dec6 = self.dec_conv6(dec5)
        upsampled_2 = self.upsample2(dec6)        
        upsampled_2 = F.adaptive_avg_pool2d(upsampled_2, (51, 51)) 
        
        conct_layer3 = torch.cat((upsampled_2, enc5), 1)  
        
        dec7 = self.dec_conv7(conct_layer3)
        dec8 = self.dec_conv8(dec7)         
        upsampled_3 = self.upsample3(dec8)
        upsampled_3 = F.adaptive_avg_pool2d(upsampled_3, (101, 101)) 
        
        conct_layer4 = torch.cat((upsampled_3, enc2), 1)                
        
        dec9 = self.dec_conv9(conct_layer4)
        dec10 = self.dec_conv10(dec9)
        dec11 = self.dec_conv11(dec10)        
        out_final = self.dec_final(dec11)
        
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
        
        train_num = int(0.9 * morphology_HR.shape[0])
        val_num = int(0.1 * morphology_HR.shape[0])
               
        if self.mode=='Train':
            morphology_HR  = morphology_HR[0:train_num]
            FE_data_HR = FE_data_HR[0:train_num]
            
        if self.mode == 'Valid':
            morphology_HR  = morphology_HR[ train_num : train_num + val_num]
            FE_data_HR = FE_data_HR[ train_num : train_num + val_num]
            
        for line in range(morphology_HR.shape[0]):
            self.morphology_HRbox.append(morphology_HR[line])
            self.FE_HRbox.append(FE_data_HR[line])          
            
    def __len__(self):
        return len(self.morphology_HRbox)

    def __getitem__(self, index):
        morph_HR = self.morphology_HRbox[index]
        morph_HR = np.array(morph_HR, dtype='float')
        morph_HR = torch.from_numpy(morph_HR)
        morph_HR = morph_HR.type(torch.FloatTensor)
        
        
        FE_HR = self.FE_HRbox[index]
        FE_HR = np.array(FE_HR, dtype='float')
        FE_HR = torch.from_numpy(FE_HR)
        FE_HR = FE_HR.type(torch.FloatTensor)
        
        return morph_HR, FE_HR


mode='Train'
batch_size_train = 50 
batch_size_valid = 50
 

if mode=='Train':
    train_dataset = MyDataset('Train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
    
    validate_dataset = MyDataset('Valid')
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size_valid)

###############################################################################
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
                for step, (morph_Hbatch, FE_HR_batch) in enumerate(tqdm(train_loader, desc=f"Training_epoch{epoch}")):
                    if torch.cuda.is_available():
                        morph_Hbatch = torch.unsqueeze(morph_Hbatch, 1)
                        FE_HR_batch = torch.unsqueeze(FE_HR_batch, 1)
                        morph_Hbatch, FE_HR_batch = morph_Hbatch.cuda(), FE_HR_batch.cuda()
                    optimizer.zero_grad()
                    prediction_origin = net(morph_Hbatch)
                    loss = MSELoss(FE_HR_batch, prediction_origin)
                    loss.backward()
                    optimizer.step()
        
                net.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for step_valid, (morph_Hvalid, FE_HR_valid) in enumerate(validate_loader):
                        if torch.cuda.is_available():
                            morph_Hvalid = torch.unsqueeze(morph_Hvalid, 1)
                            FE_HR_valid = torch.unsqueeze(FE_HR_valid, 1)
                            
                            morph_Hvalid, FE_HR_valid = morph_Hvalid.cuda(), FE_HR_valid.cuda()      
                        prediction_validate = net(morph_Hvalid)
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
                    f.write(f"iter_num: {epoch + 1}      loss: {loss.item():.8f}    loss_validate: {loss_validate.item():.8f}  \r\n")
                    torch.save(net.state_dict(), f'{dir_name}/net_{epoch + 1}_epoch.pkl')
                    torch.save(optimizer.state_dict(), f'{dir_name}/optimizer_{epoch + 1}_epoch.pkl')
                    loss_plot.append(loss.item())
                    loss_validate_plot.append(loss_validate.item())
                
                average_val_loss = total_val_loss / len(validate_loader)
                #scheduler.step(average_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}/{n_epochs}, Training Loss: {loss.item():.8f}, Validation Loss: {average_val_loss:.8f}, Learning Rate: {current_lr}")
        
            best_epoch = (loss_validate_plot.index(min(loss_validate_plot)) + 1) 
            print(f"The ANN has been trained, the best epoch is {best_epoch}")
            
################################################################################
morph_HR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\test_morph_HR.txt") 
FE_HR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\output_FE_test_HR.txt") 
results = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\Output_pre-trained_case02_test.txt")
results = split_data(results,101)

net.load_state_dict(torch.load(r"K:\Najafi\codes\UNet_SH\Unet Classic\net_483_epoch.pkl"))
optimizer.load_state_dict(torch.load(r"K:\Najafi\codes\UNet_SH\Unet Classic\optimizer_483_epoch.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)   
net.eval()

os.chdir(r"K:\Najafi\codes\UNet_SH\Unet Classic")
with PdfPages('Flux_UNet_classic.pdf') as pdf:
    for i in range(len(morph_HR)):
        morphology = morph_HR[i]
        morphology = np.expand_dims(morphology, axis=0)
        morphology = torch.from_numpy(morphology.astype(np.float32))
        morphology = morphology.type(torch.FloatTensor)
        morphology = morphology.to(device)         
        
        FE_L = FE_LR[i]
        FE_L = np.expand_dims(FE_L, axis=0)
        FE_L = torch.from_numpy(FE_L.astype(np.float32))
        FE_L = FE_L.type(torch.FloatTensor)
        FE_L = FE_L.to(device)           
        
        FE_H = FE_HR[i]
        FE_H = torch.from_numpy(FE_H.astype(np.float32))
        FE_H = FE_H.to(device)       
        test = net(morphology)
        
        N_HR = 101
        Ne = N_HR-1
        L = 1.0  # Length of the square domain
        dx = L / Ne  # grid spacing
        
        xx = np.linspace(0, L, N_HR)
        yy = np.linspace(0, L, N_HR)
        X, Y = np.meshgrid(xx, yy)

        conductivity_map = morphology.cpu().data.numpy()
        T_mesh1 = test.cpu().data.numpy()
        T_mesh2 = FE_H.cpu().data.numpy()
        
        dT_dx_HR1 = -conductivity_map[0][0] * np.gradient(T_mesh1[0][0], dx, axis=1)  # Heat flux in x-direction
        dT_dy_HR1 = -conductivity_map[0][0] * np.gradient(T_mesh1[0][0], dx, axis=0) 
        
        dT_dx_HR2 = -conductivity_map[0][0] * np.gradient(T_mesh2[0], dx, axis=1)  # Heat flux in x-direction
        dT_dy_HR2 = -conductivity_map[0][0] * np.gradient(T_mesh2[0], dx, axis=0)#  Heat flux in y-direction
                
        flux_magnitude_HR1 = np.sqrt(dT_dx_HR1**2 + dT_dy_HR1**2)
        flux_magnitude_HR2 = np.sqrt(dT_dx_HR2**2 + dT_dy_HR2**2)
        
        fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))                   
        titles = ['Morphology', 'UNet: test', 'GT: FE', '(GT - UNet)']
        data = [
            morphology[0, 0, :, :].cpu().data.numpy(),
            flux_magnitude_HR1,
            flux_magnitude_HR2,
            np.abs(FE_H[0].cpu().data.numpy() - test[0, 0, :, :].cpu().data.numpy())
               
        ]
        colormaps = ['viridis', 'hot', 'hot', 'hot', 'hot'] 
        np.lins
        
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
        
        
with PdfPages('UNet_SH_classic.pdf') as pdf:
    for i in range(len(morph_HR)):
        morphology = morph_HR[i]
        morphology = np.expand_dims(morphology, axis=0)
        morphology = torch.from_numpy(morphology.astype(np.float32))
        morphology = morphology.type(torch.FloatTensor)
        morphology = morphology.to(device) 
               
        FE_H = FE_HR[i]
        FE_H = torch.from_numpy(FE_H.astype(np.float32))
        FE_H = FE_H.to(device)
        FE = Variable((FE).to(device))        
        test = net(morphology)
                
        fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))  
        
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
            ax.set_xticks([])  
            ax.set_yticks([])  
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=30)  # 
        
        plt.tight_layout()
        pdf.savefig(fig)  
        plt.close(fig)  
    
