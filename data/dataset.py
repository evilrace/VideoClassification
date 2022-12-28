import torch
import numpy as np
import pandas as pd
import cv2
import torchvision
data_path = r'F:\dataset\ucf101_top5'

train_data_csv = pd.read_csv(f'{data_path}\\train.csv')
test_data_csv = pd.read_csv(f'{data_path}\\test.csv')
train_data_csv = train_data_csv.sample(frac=1).reset_index()
test_data_csv = test_data_csv.sample(frac=1).reset_index()

def path_deco(type):
    def path_preprocess(data):
        data['video_name'] = f'{data_path}\\{type}\\{data["video_name"]}'
        return data
    return path_preprocess
train_path_preprocess = path_deco('train')
test_path_preprocess = path_deco('test')
train_data_csv = train_data_csv.apply(train_path_preprocess, axis=1)
test_data_csv = test_data_csv.apply(test_path_preprocess, axis=1)

def get_features(features, name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, data_type) -> None:
        super().__init__()
        self.features = {}
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to('cuda')
        self.hook = get_features(self.features, 'fc')
        self.resnet.avgpool.register_forward_hook(self.hook)
        self.resnet.eval()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x : x/255.),
        ]
        )

        self.seq_num = 20
        if data_type == 'train':
            self.data_csv = train_data_csv
        elif data_type == 'test':
            self.data_csv = test_data_csv
        self.labels = list(set(self.data_csv['tag']))
        self.cached = {}

    def __getitem__(self, idx):
        if idx not in self.cached:
            img_path = self.data_csv.iloc[idx]['video_name']
            label = self.data_csv.iloc[idx]['tag']
            frame,_,_ = torchvision.io.read_video(img_path)
            with torch.no_grad():
                frame = frame.permute(0,3,1,2).to('cuda')
                frame = self.transforms(frame[::10])
                frame = frame[:20,:]
                frame = frame.float()    
                self.resnet(frame)
                output_features = self.features['fc']
                output_features = output_features.to('cpu')
                output_features = torch.reshape(output_features,(output_features.shape[0],output_features.shape[1]))
                self.cached[idx] = (output_features,label)
                
        return self.cached[idx]
    def __len__(self):
        return self.data_csv.shape[0]

    def get_label(self):
        return self.labels


