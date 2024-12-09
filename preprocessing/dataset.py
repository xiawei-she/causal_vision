import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torch



class CarDataset(Dataset):

    def __init__(self,data_dir):
        self.data_dir=data_dir

        self.data_transforms = transforms.Compose([transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)
                                                 , transforms.RandomHorizontalFlip(0.1)
                                                 , transforms.RandomVerticalFlip(0.1)
                                                 , transforms.ToTensor()])

        with open(os.path.join(self.data_dir,'labels.txt'), 'r', encoding='utf-8') as file:
            content = file.read()
        label_dict = dict()
        for item in content.split('\n'):
            if len(item.split(' '))>1:
                idx,label = item.split(' ')[0],item.split(' ')[1]
                if idx not in label_dict.keys():
                    label_dict[int(idx)]=label
        self.label_dict = label_dict
        self.labels = list(set(label_dict.values()))

    def __getitem__(self, idx):
            image_path = os.path.join(self.data_dir, str(idx)+'.jpg')
            img = Image.open(image_path)
            img = self.data_transforms(img)

            label = torch.tensor(float(self.label_dict[idx]), dtype=torch.float32)
            return img, label

    def __len__(self):
        return len(self.label_dict)


