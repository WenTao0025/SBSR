import os

import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from configs.Vit.config import config_argument


arg = config_argument()
class SketchNet(nn.Module):
    def __init__(self):
        super(SketchNet, self).__init__()
        self.vit = vit_l_16(weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    def forward(self,x):
        x = self.vit(x)
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        return x1,x2
class ModelNet(nn.Module):
    def __init__(self):
        super(ModelNet, self).__init__()
        self.vit = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1)
    def forward(self,x):
        x = x.reshape(-1,arg.channels,arg.img_size,arg.img_size)
        x = self.vit(x)
        embding = x.size()[-1]
        x = x.reshape(-1, arg.num_views, embding)
        x = x.mean(dim=1)
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)

        return x1,x2
class CrossDomainNet(nn.Module):
    def __init__(self):
        super(CrossDomainNet, self).__init__()
        self.QueryNet = SketchNet()
        self.ModelNet = ModelNet()
        self.mlp1 = nn.Sequential(nn.Linear(1000, arg.embding_size),
                                  nn.BatchNorm1d(arg.embding_size),
                                  )
        self.mlp2 = nn.Sequential(nn.Linear(arg.embding_size, arg.class_num))
    def forward(self,query,render):
        sketch_embding = self.QueryNet.vit(query)
        model_embding = self.ModelNet.vit(render)
        embding = model_embding.size()[-1]
        model_embding = model_embding.reshape(-1, arg.num_views, embding)
        model_embding = model_embding.mean(dim=1)

        #经过VIT后得到草图和模型的向量

        query_emb1 = self.mlp1(sketch_embding)
        query_emb1 = torch.nn.functional.normalize(query_emb1,p=2,dim=1)
        query_emb2 = self.mlp2(query_emb1)

        model_embding1 = self.mlp1(model_embding)
        model_embding1 = torch.nn.functional.normalize(model_embding1,p=2,dim=1)
        model_embding2 = self.mlp2(model_embding1)

        return query_emb1,query_emb2,model_embding1,model_embding2
    def get_render_emb(self,render):
        model_embding = self.ModelNet.vit(render)
        embding = model_embding.size()[-1]
        model_embding = model_embding.reshape(-1, arg.num_views, embding)
        model_embding = model_embding.mean(dim=1)
        model_embding1 = self.mlp1(model_embding)
        model_embding1 = torch.nn.functional.normalize(model_embding1, p=2, dim=1)
        return model_embding1
    def get_sketch_emb(self,query):
        sketch_embding = self.QueryNet.vit(query)
        sketch_embding1 = self.mlp1(sketch_embding)
        sketch_embding1 = torch.nn.functional.normalize(sketch_embding1, p=2, dim=1)
        return sketch_embding1



if __name__ == '__main__':
    x = torch.randn(16,12,3,224,224)
    a = ModelNet()
    i1,i2 = a(x)
    print(i1.shape)
    print(i2.shape)





