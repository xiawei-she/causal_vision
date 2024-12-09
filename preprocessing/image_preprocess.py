import os
from PIL import Image



# def preprocess(img,img_size):
    # return img.resize(img_size,Image.ANTIALIAS)


if __name__ == '__main__':
    directory = 'E:/Projet_CausalVision/images/damaged' #normal damaged
    data_path = '../data/test'
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory,filename))
        print(f"img {filename}: size {img.size},format {img.format}")
        img= img.convert('RGB')
        img_pre = img.resize((1024,1024),resample=Image.BILINEAR)
        img_pre.save(data_path+"/"+filename.split('.')[0]+".jpg")