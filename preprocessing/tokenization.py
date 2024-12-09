import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image


class Tokenize(nn.Module):
    def __init__(self,img_h,img_w,in_channels,seg_h,seg_w,token_len):

        super(Tokenize, self).__init__()
        self.img_len = img_h
        self.img_wid = img_w
        self.in_channels = in_channels
        self.seg_h = seg_h
        self.seg_w = seg_w
        assert img_h % seg_h == 0 and img_w % seg_w == 0,"img size must be divisible by seg_h and seg_w"

        self.ntoken = (img_h /seg_h)*(img_w /seg_w)
        self.token_len = token_len
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=token_len,
                              stride=seg_w,
                              kernel_size=(seg_h,seg_w))
    def _trainable_position_embedding(self,n_token,token_len):

        position_embedding = nn.Embedding(n_token,token_len)
        nn.init.constant_(position_embedding.weight,0)

        return position_embedding
    def forward(self, image,label):
        # input image shape = [batch,channels,height,width]
        image_token = self.conv(image)
        image_token = image_token.reshape(image_token.size(0),image_token.size(1), -1)

        return image_token


if __name__ == '__main__':
    data_path = '../data/normal'
    filename = '1.jpg'
    img = Image.open(os.path.join(data_path, filename))
    img = torch.from_numpy(np.array(img))
    img = img.unsqueeze(0).permute(0,3,1,2).to(torch.float32)
    print(img.shape)

    tokenizor = Tokenize(img_h=1024,img_w=1024,in_channels=3,seg_h=128,seg_w=128,token_len=16)
    img_token =tokenizor(img)
    print(img_token.shape)

    norm1 = nn.LayerNorm(16)
    img_token = norm1(img_token)
    print(img_token.shape)