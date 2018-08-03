import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms

class DoubleConv(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(output_channels,output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class SingleConv(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(SingleConv,self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.single_conv(x)
        return x


class DoubleConv2(nn.Module):
    def __init__(self,input_channels, middle_channels, output_channels):
        super(DoubleConv2,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, middle_channels,3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(True),
            nn.Conv2d(middle_channels,output_channels,3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownConv,self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(input_channels, output_channels)
        )

    def forward(self,x):
        x = self.down_conv(x)
        return x



class DownConv2(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels):
        super(DownConv,self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv2(input_channels,middle_channels, output_channels)
        )

    def forward(self,x):
        x = self.down_conv(x)
        return x



class UpConv(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(UpConv,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self,x, x_prev):
        x = self.up_conv(x)
        x = torch.cat((x, x_prev), 1)
        x = self.conv(x)
        return x

class SimpleUpConv(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(SimpleUpConv,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = SingleConv(input_channels, output_channels)

    def forward(self,x, x_prev):
        x = self.up_conv(x)
        x = self.conv(x)
        return x




class DownSpatial(nn.Module):
    def __init__(self, input_channels, output_channels, H_in, W_in, depth, dropout = 0.5):
        super(DownSpatial,self).__init__()
        self.down_spatial = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(input_channels, output_channels),
            SpatialTransformer(output_channels, H_in, W_in, depth, dropout= dropout),
            DoubleConv(output_channels, output_channels),
        )

    def forward(self,x):
        x = self.down_spatial(x)
        return x



class UpSpatial(nn.Module):
    def __init__(self, input_channels,output_channels, H_in, W_in, depth, dropout = 0.5):
        super(UpSpatial,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = DoubleConv(input_channels, output_channels)
        self.spatial = SpatialTransformer(input_channels//2,H_in,W_in, depth, dropout =dropout )

    def forward(self,x, x_prev):
        x = self.up_conv(x)
        x_prev = self.spatial(x_prev)
        x = torch.cat((x, x_prev), 1)
        x = self.conv(x)
        return x





class ConvPool(nn.Module):
    def __init__(self, input_channels,output_channels):

        super(ConvPool,self).__init__()
        self.conv_pool = nn.Sequential(
        DoubleConv(input_channels, output_channels),
        nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.conv_pool(x)








class LocalisationNetwork(nn.Module):
    def __init__(self, channels_in,H_in,W_in, depth, dropout = 0.5):
        super(LocalisationNetwork,self).__init__()
        # Spatial transformer localization-network

        layers = []
        layers.append(ConvPool(channels_in, channels_in //2))
        channels = channels_in//2
        H_in = H_in //2
        W_in = W_in //2
        for i in range(1,depth):
            layers.append(ConvPool(channels,channels//2))
            channels = channels //2
            H_in = H_in //2
            W_in = W_in //2


        self.localization = nn.Sequential(*layers)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(  # VERIFY THIS
            nn.Linear(channels*H_in*W_in, 64),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.channels = channels
        self.H_out = H_in
        self.W_out = W_in

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.channels*self.H_out*self.W_out)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        return theta








class SpatialTransformer(nn.Module):

    def __init__(self, channels_in,H_in,W_in, depth, dropout = 0.5):
        super(SpatialTransformer,self).__init__()
        # Spatial transformer localization-network

        self.localisation_network = LocalisationNetwork( channels_in,H_in,W_in, depth, dropout =dropout)

    def forward(self, x):
        theta = self.localisation_network(x)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x




class AffineTransformer(nn.Module):
    def __init__(self):
        super(AffineTransformer,self).__init__()

    def forward(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x



class UpAffine(nn.Module):
    def __init__(self, input_channels,middle_channels, output_channels):
        super(UpAffine,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = DoubleConv2(int(3*input_channels//2), middle_channels,output_channels)
        self.spatial = AffineTransformer()

    def forward(self,x, x_prev,theta):
        x = self.up_conv(x)
        x_trans = self.spatial(x_prev, theta)
        x = torch.cat((x, x_trans, x_prev), 1)
        x = self.conv(x)
        return x

class UpAffineBaseline(nn.Module):
    def __init__(self, input_channels):
        super(UpAffineBaseline,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.spatial = AffineTransformer()

    def forward(self,x, x_prev,theta):
        x = self.up_conv(x)
        x_trans = self.spatial(x_prev, theta)
        x = torch.cat((x, x_trans, x_prev), 1)
        return x



class UpAffineWithRelu(nn.Module):
    def __init__(self, input_channels,middle_channels, output_channels):
        super(UpAffineWithRelu,self).__init__()
        self.up_conv = nn.Sequential( nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2),
                                      nn.BatchNorm2d(input_channels//2),
                                      nn.ReLU(True),
                                      )
        self.conv = DoubleConv2(int(3*input_channels//2), middle_channels,output_channels)
        self.spatial = AffineTransformer()

    def forward(self,x, x_prev,theta):
        x = self.up_conv(x)
        x_trans = self.spatial(x_prev, theta)
        x = torch.cat((x, x_trans, x_prev), 1)
        x = self.conv(x)
        return x

class UpAffineBaselineWithRelu(nn.Module):
    def __init__(self, input_channels):
        super(UpAffineBaselineWithRelu,self).__init__()
        self.up_conv = nn.Sequential( nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2),
                                      nn.BatchNorm2d(input_channels//2),
                                      nn.ReLU(True),
                                      )
        self.spatial = AffineTransformer()

    def forward(self,x, x_prev,theta):
        x = self.up_conv(x)
        x_trans = self.spatial(x_prev, theta)
        x = torch.cat((x, x_trans, x_prev), 1)
        return x



class SpatialConv(nn.Module):
    def __init__(self, input_channels, H_in, W_in, depth, dropout = 0.5):
        super(SpatialConv,self).__init__()
        self.local = LocalisationNetwork(input_channels, H_in, W_in, depth, dropout= dropout)
        self.affine = AffineTransformer()
        self.conv = DoubleConv(int(input_channels*2),input_channels)

    def forward(self,x):
        theta = self.local(x)
        x_trans = self.affine(x, theta)
        x_final = torch.cat((x_trans, x), 1)
        x_final = self.conv(x_final)
        return x_final, theta



# Uncompatible with binary cross-entropy loss
class SpatialUnet2(nn.Module):
    def __init__(self,initial_channels, initial_h = 128, initial_w = 256, dropout = 0.5, wrap_input_mask = True, starting_channels = 64, extra_relu = False, adjust_transform =False):
        super(SpatialUnet2, self).__init__()

        self.wrap_input_mask = wrap_input_mask
        self.adjust_transform = adjust_transform
        self.initial_conv = DoubleConv(initial_channels,starting_channels) #128*256*64
        self.down1 = DownConv(starting_channels,starting_channels*2)  #64*128*128
        self.down2 = DownConv(starting_channels*2,starting_channels*4) #32*64*256
        self.down3 = DownConv(starting_channels*4,starting_channels*8) #16*32*512
        self.bottle_neck_spatial = SpatialConv(starting_channels*8,initial_h//8,initial_w//8, 3, dropout) #16*32*1024

        if(adjust_transform):
            self.multiplier1 = torch.ones((1,2,3),requires_grad = True)
            self.multiplier2 = torch.ones((1,2,3),requires_grad = True)
            self.multiplier3 = torch.ones((1,2,3),requires_grad = True)

        if(extra_relu):
            self.up2 = UpAffineWithRelu(starting_channels*8,starting_channels*8, starting_channels*4) #32*64*256
            self.up3 = UpAffineWithRelu( starting_channels*4, starting_channels*4, starting_channels*2)  #64*128 *128
        else:
            self.up2 = UpAffine(starting_channels*8,starting_channels*8, starting_channels*4) #32*64*256
            self.up3 = UpAffine( starting_channels*4, starting_channels*4, starting_channels*2)  #64*128 *128
        if(wrap_input_mask):
            if(extra_relu):
                self.up4 = UpAffineBaselineWithRelu(starting_channels*2) #128*256 *192
            else:
                self.up4 = UpAffineBaseline(starting_channels*2) #128*256 *192
            self.mask_transformer = AffineTransformer()
            self.double_conv = DoubleConv2(starting_channels*3+1,starting_channels*2,starting_channels)
            self.single_conv = SingleConv(starting_channels,starting_channels//2)
        else:
            if(extra_relu):
                self.up4 = UpAffineWithRelu(starting_channels*2,starting_channels*2,starting_channels)
            else:
                self.up4 = UpAffine(starting_channels*2,starting_channels*2,starting_channels)
            self.double_conv = SingleConv(starting_channels,starting_channels//2)

        self.final_conv = nn.Conv2d(starting_channels//2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x) #128*256*64
        x2 = self.down1(x1) #64*128*128
        x3 = self.down2(x2) #32*64*256
        x4 = self.down3(x3) #16*32*512
        x5, theta = self.bottle_neck_spatial(x4) #16*32*1024

        if(self.adjust_transform):
            theta1 = torch.mul(theta,self.multiplier1)
            theta2 = torch.mul(theta,self.multiplier2)
            theta3 = torch.mul(theta,self.multiplier3)
            x6 = self.up2(x5,x3, theta1)  #32*64*512
            x6 = self.up3(x6,x2,theta2) # 32*128*256
            x6 = self.up4(x6,x1, theta3) #64*128*256 (or 128)
        else:

            x6 = self.up2(x5,x3, theta)  #32*64*512
            x6 = self.up3(x6,x2,theta) # 32*128*256
            x6 = self.up4(x6,x1, theta) #64*128*256 (or 128)

        if(self.wrap_input_mask):
            mask = x[:,9,:,:]
            mask = mask.unsqueeze(1)
            mask_translated = self.mask_transformer(mask,theta)

            x = torch.cat((x6,mask_translated),1) #64*128*257
            x = self.double_conv(x)
            x = self.single_conv(x)
        else:
            x = self.double_conv(x6)


        x = self.final_conv(x)


        if(self.wrap_input_mask):
            return x, mask_translated
        else:
            return x


    def eval_forward(self,x):
        if(self.wrap_input_mask):
            x,mask_translated = self.forward(x)
            return self.sigmoid(x), mask_translated
        else:
            return self.sigmoid(self.forward(x))

    def forward_mask(self,x):

        if(self.wrap_input_mask):
            x,_ = self.eval_forward(x)
        else:
            x = self.eval_forward(x)

        return torch.ge(x,0.5)









class SpatialUnet(nn.Module):
    def __init__(self,initial_channels, initial_h = 128, initial_w = 256, dropout = 0.5):
        super(SpatialUnet, self).__init__()

        depth = int(4 - (np.log2(128) - np.log2(initial_h)))
        self.initial_conv = DoubleConv(initial_channels,64) #128*256*64
        self.down1 = DownConv(64,128)  #64*128*128
        self.down2 = DownConv(128,256) #32*64*256
        self.down3 = DownSpatial(256,512,initial_h//8,initial_w//8, depth-1, dropout) #16*32*512

        self.up2 = UpSpatial(512,256, initial_h//4, initial_w//4, depth= depth, dropout = dropout) #32*64*256
        self.up3 = UpSpatial(256,128, initial_h//2, initial_w//2, depth= depth, dropout = dropout)  #64*128 *128
        self.up4 = UpSpatial(128,64, initial_h, initial_w, depth = depth, dropout = dropout) #128*256 *64
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final_conv(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,0.5)





class SingleConvAvgPool(nn.Module):
    def __init__(self, input_channels,output_channels):

        super(SingleConvAvgPool,self).__init__()
        self.single_conv_pool = nn.Sequential(
        nn.Conv2d(input_channels, output_channels,3, padding=1),
        nn.AvgPool2d(2, stride=2),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True)
        )

    def forward(self, x):
        return self.single_conv_pool(x)

class SingleConvAvgPool2(nn.Module):
    def __init__(self, input_channels,output_channels):

        super(SingleConvAvgPool2,self).__init__()
        self.single_conv_pool = nn.Sequential(
        nn.Conv2d(input_channels, output_channels,3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True),
        nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.single_conv_pool(x)





class ResnetFeatureExtractor(nn.Module):
    def __init__(self, mask_channels, correction = False):
        super(ResnetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.resnet_fixed = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu, resnet.maxpool,  resnet.layer1, resnet.layer2)

        for param in self.resnet_fixed.parameters():
            param.requires_grad = False

        if(correction):
            self.triple_down_sample = nn.Sequential(
                SingleConvAvgPool2(mask_channels,16),
                SingleConvAvgPool2(16,32),
                SingleConvAvgPool2(32,64)
            )
        else:
            self.triple_down_sample = nn.Sequential(
                SingleConvAvgPool(mask_channels,16),
                SingleConvAvgPool(16,32),
                SingleConvAvgPool(32,64)
            )



    def forward(self, x): #TODO: Check that I'm getting the images correctly
        rgb1 = x[:,0:3,:,:]
        rgb2 = x[:,3:6,:,:]
        rgb3 = x[:,6:9,:,:]
        masks = x[:,9:,:,:]

        if(len(masks.size())<4):
            masks = masks.unsqueeze(1)


        rgb1 = self.resnet_fixed(rgb1)
        rgb2 = self.resnet_fixed(rgb2)
        rgb3 = self.resnet_fixed(rgb3)

        masks = self.triple_down_sample(masks)

        images = torch.cat((rgb1, rgb2,rgb3, masks), 1)

        return images


# Uncompatible with binary cross-entropy loss
class SpatialUnet2SM(nn.Module):
    def __init__(self,initial_channels, initial_h = 32, initial_w = 64, dropout = 0.5, starting_channels = 64, final_upscaling = 4):
        super(SpatialUnet2SM, self).__init__()


        self.initial_conv = DoubleConv2(initial_channels,initial_channels//2, starting_channels) #64*128*32
        self.down1 = DownConv(starting_channels,starting_channels*2)  #64*128*64
        self.down2 = DownConv(starting_channels*2,starting_channels*4) #32*64*128
        self.bottle_neck_spatial = SpatialConv(starting_channels*4,initial_h//4,initial_w//4, 2, dropout) #16*32*128


        self.up2 = UpAffine(starting_channels*4,starting_channels*4, starting_channels*2) #32*64*64

        self.up3 = UpAffine( starting_channels*2, starting_channels*2, starting_channels)  #64*128 *32
        self.double_conv = SingleConv(starting_channels,starting_channels//2)

        self.final_conv = nn.Conv2d(starting_channels//2, 1, 1)

        self.upsample = nn.Upsample(scale_factor=final_upscaling,mode = 'bilinear')


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x) #128*256*32
        x2 = self.down1(x1) #64*128*64
        x3 = self.down2(x2) #32*64*128
        x4, theta = self.bottle_neck_spatial(x3) #16*32*128


        x5 = self.up2(x4,x2, theta)  #32*64*64
        x5 = self.up3(x5,x1,theta) # 32*128*32
        x = self.double_conv(x5)


        x = self.final_conv(x)

        x = self.upsample(x)



        return x


    def eval_forward(self,x):

        return self.sigmoid(self.forward(x))

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,0.5)










class SpatialUNetOnFeatures(nn.Module):
    def __init__(self, input_channels, final_upscaling = 4, correction = False):
        super(SpatialUNetOnFeatures, self).__init__()

        self.spatial_unet = SpatialUnet2SM(448, final_upscaling= final_upscaling)
        if(input_channels == 12):
            self.feature_extractor = ResnetFeatureExtractor(3, correction)
        elif(input_channels == 10):
            self.feature_extractor = ResnetFeatureExtractor(1, correction)
        else:
            raise ValueError('Incorrect number of input channels')


    def forward(self, x):
        x = self.feature_extractor(x)
        return self.spatial_unet(x)

    def eval_forward(self,x):
        x = self.feature_extractor(x)
        return self.spatial_unet.eval_forward(x)

    def forward_mask(self,x):
        x = self.feature_extractor(x)
        return self.spatial_unet.forward_mask(x)













class UnetShallowSM(nn.Module):
    def __init__(self,initial_channels,starting_depth=64, final_upscaling = 4):
        super(UnetShallowSM, self).__init__()
        self.initial_conv = DoubleConv(initial_channels,starting_depth) #64*128*32
        self.down1 = DownConv(starting_depth,starting_depth*2) #32*64*64
        self.down2 = DownConv(starting_depth*2,starting_depth*4) #16*32*128
        self.up2 = UpConv(starting_depth*4,starting_depth*2) #32*64*64
        self.up3 = UpConv(starting_depth*2,starting_depth) #64*128*32
        self.final_conv = nn.Conv2d(starting_depth, 1, 1)
        self.upsample = nn.Upsample(scale_factor=final_upscaling,mode = 'bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up2(x3,x2)
        x = self.up3(x,x1)
        x = self.final_conv(x)
        x = self.upsample(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,0.5)








class UNetOnFeatures(nn.Module):
    def __init__(self, input_channels, final_upscaling=4, correction = False):
        super(UNetOnFeatures, self).__init__()

        self.unet = UnetShallowSM(448, final_upscaling = final_upscaling)
        if(input_channels == 12):
            self.feature_extractor = ResnetFeatureExtractor(3, correction)
        elif(input_channels == 10):
            self.feature_extractor = ResnetFeatureExtractor(1, correction)
        else:
            raise ValueError('Incorrect number of input channels')


    def forward(self, x):
        x = self.feature_extractor(x)
        return self.unet(x)

    def eval_forward(self,x):
        x = self.feature_extractor(x)
        return self.unet.eval_forward(x)

    def forward_mask(self,x):
        x = self.feature_extractor(x)
        return self.unet.forward_mask(x)









class Unet(nn.Module):
    def __init__(self,initial_channels):
        super(Unet, self).__init__()
        self.initial_conv = DoubleConv(initial_channels,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.down4 = DownConv(512,1024)
        self.up1= UpConv(1024,512)
        self.up2= UpConv(512,256)
        self.up3= UpConv(256,128)
        self.up4= UpConv(128,64)
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final_conv(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,torch.tensor(0.5))






class UnetShallow(nn.Module):
    def __init__(self,initial_channels):
        super(UnetShallow, self).__init__()
        self.initial_conv = DoubleConv(initial_channels,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.up2 = UpConv(512,256)
        self.up3 = UpConv(256,128)
        self.up4 = UpConv(128,64)
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final_conv(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,0.5)








################## LEGACY SPATIAL UNET ###############################################################################


class SingleConvPool(nn.Module):
    def __init__(self, input_channels,output_channels):

        super(SingleConvPool,self).__init__()
        self.single_conv_pool = nn.Sequential(
        nn.Conv2d(input_channels, output_channels,3, padding=1),
        nn.MaxPool2d(2, stride=2),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True)
        )

    def forward(self, x):
        return self.single_conv_pool(x)




class SpatialTransformer0(nn.Module):

    def __init__(self, channels_in,H_in,W_in, depth, dropout = 0.5):
        super(SpatialTransformer0,self).__init__()
        # Spatial transformer localization-network

        layers = []
        layers.append(SingleConvPool(channels_in, channels_in //2))
        channels = channels_in//2
        H_in = H_in //2
        W_in = W_in //2
        for i in range(1,depth):
            layers.append(SingleConvPool(channels,channels//2))
            channels = channels //2
            H_in = H_in //2
            W_in = W_in //2


        self.localization = nn.Sequential(*layers)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Dropout(p = dropout),   # VERIFY THIS
            nn.Linear(channels*H_in*W_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.channels = channels
        self.H_out = H_in
        self.W_out = W_in

        # Initialize the weights/bias with identity transformation
        self.fc_loc[7].weight.data.zero_()
        self.fc_loc[7].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.channels*self.H_out*self.W_out)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x



class DownSpatial0(nn.Module):
    def __init__(self, input_channels, output_channels, H_in, W_in, depth, dropout = 0.5):
        super(DownSpatial0,self).__init__()
        self.down_spatial = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(input_channels, output_channels),
            SpatialTransformer0(output_channels, H_in, W_in, depth),
            DoubleConv(output_channels, output_channels),
        )

    def forward(self,x):
        x = self.down_spatial(x)
        return x



class UpSpatial0(nn.Module):
    def __init__(self, input_channels,output_channels, H_in, W_in, depth, dropout = 0.5):
        super(UpSpatial0,self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_channels,input_channels//2, 2, stride =2)
        self.conv = DoubleConv(input_channels, output_channels)
        self.spatial = SpatialTransformer0(input_channels//2,H_in,W_in, depth, dropout =dropout )

    def forward(self,x, x_prev):
        x = self.up_conv(x)
        x_prev = self.spatial(x_prev)
        x = torch.cat((x, x_prev), 1)
        x = self.conv(x)
        return x











class SpatialUnet0(nn.Module):
    def __init__(self,initial_channels, initial_h = 128, initial_w = 256, dropout = 0.5):
        super(SpatialUnet0, self).__init__()

        depth = int(4 - (np.log2(128) - np.log2(initial_h)))
        self.initial_conv = DoubleConv(initial_channels,64) #128*256*64
        self.down1 = DownConv(64,128)  #64*128*128
        self.down2 = DownConv(128,256) #32*64*256
        self.down3 = DownSpatial0(256,512,initial_h//8,initial_w//8, depth-1, dropout) #16*32*512

        self.up2 = UpSpatial0(512,256, initial_h//4, initial_w//4, depth= depth, dropout = dropout) #32*64*256
        self.up3 = UpSpatial0(256,128, initial_h//2, initial_w//2, depth= depth, dropout = dropout)  #64*128 *128
        self.up4 = UpSpatial0(128,64, initial_h, initial_w, depth = depth, dropout = dropout) #128*256 *64
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.final_conv(x)
        return x

    def eval_forward(self,x):
        x = self.forward(x)
        return self.sigmoid(x)

    def forward_mask(self,x):
        x = self.eval_forward(x)

        return torch.ge(x,0.5)




########################################################################################################################

