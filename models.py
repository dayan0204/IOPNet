import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import timm
from timm.models.vision_transformer import Block
from swin import SwinTransformer
from einops import rearrange

class Unpool(nn.Module):

    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride


        self.weights = torch.zeros(num_channels, 1, stride, stride)
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights.to(x), stride=self.stride, groups=self.num_channels)

def weights_init(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):


    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

class UpConv(Decoder):

    def upconv_module(self, in_channels):

        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):


    class UpProjModule(nn.Module):


        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):

    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']


        del pretrained_model


        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)


        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)


        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(num_channels // 2, 512, bias=True)
        self.fc2 = nn.Linear(512, 19, bias=True)
        self.drop = nn.Dropout(0.15)
        self.globalpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.globalpooling(x)
        x1 = x.view(x.size(0),-1)

        x1 = self.leakyrelu(self.fc1(x1))
        x1 = self.drop(x1)
        x1 = self.relu(self.fc2(x1))


        return x1

from torchvision import transforms, utils, models
class VGGModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=True, load_weight=1):
        super(VGGModel, self).__init__()

        self.num_channels = num_channels
        self.vgg = models.vgg16(pretrained=bool(load_weight)).features
        
        for param in self.vgg.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = self.vgg[:7]
        self.conv_layer2 = self.vgg[7:12]
        self.conv_layer3 = self.vgg[12:19]
        self.conv_layer4 = self.vgg[19:24]
        self.conv_layer5 = self.vgg[24:]

        self.conv_layer11 = self.vgg[:7]
        self.conv_layer22 = self.vgg[7:12]
        self.conv_layer33 = self.vgg[12:19]
        self.conv_layer44 = self.vgg[19:24]
        self.conv_layer55 = self.vgg[24:]

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.globalpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv6 = nn.Conv2d(in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=True)
        self.conv6.bias.data.fill_(1.)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(256, 56, bias=True)
        self.fc2 = nn.Linear(56, 19, bias=True)
        self.drop = nn.Dropout(0.4)
        self.fc_score = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(80, 40),
            nn.ReLU()
        )


        self.pooling = nn.AvgPool2d(2,stride=2)
        self.pooling1 = nn.MaxPool2d(2,stride=2)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def decode_ord(self, y):
        height = 1
        width = 1
        batch_size, prob = y.shape
        y = torch.reshape(y, (batch_size, prob//2, 2, height, width))
        denominator = torch.sum(torch.exp(y), 2)
        pred_score = torch.div(torch.exp(y[:, :, 1, :, :]), denominator)
        return pred_score


        


    
    def forward(self, images):

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)

        out3 = self.conv_layer3(out2)

        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        x1 = self.conv6(out5)

        x1 = self.globalpooling(x1)


        x1 = x1.view(x1.size(0),-1)
        x1 = self.leakyrelu(self.fc1(x1))
        x1 = self.drop(x1)
        x1 = self.relu(self.fc2(x1))


        return x1


import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from swin import SwinTransformer
from torch import nn
from einops import rearrange


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, pretrained=True, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=pretrained)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)


        self.reduce_channels = nn.Conv1d(3072, 288, kernel_size=1)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score = nn.Sequential(
            nn.Linear(28224, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, 19),
            nn.ReLU()
        )

        self.fc_weight = nn.Sequential(
            nn.Linear(28224, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, 19),
            nn.Sigmoid()
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        self.save_output.clear()
        try:
            self.vit.forward_features(x)
            x = self.extract_feature(self.save_output)
        finally:
            self.save_output.clear()


        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)


        x = self.reduce_channels(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)


        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)


        x = x.view(x.size(0), -1)


        f = self.fc_score(x)
        w = self.fc_weight(x)




        score = f * w


        return score
