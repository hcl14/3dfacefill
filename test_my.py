# main.py

import sys
import traceback
import torch
import torch.nn as nn
import random
import config
import utils
from model import Model
from train import Trainer
from test import Tester

#from dataloader import Dataloader
from models import UnetSeg
import faulthandler
faulthandler.enable()


import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
#import albumentations
#from albumentations.pytorch import ToTensor
from torch.utils.data.distributed import DistributedSampler

import os, glob

import numpy as np
import PIL.Image

from torch.utils.data import Dataset
import torchvision

from tqdm import tqdm

# pip install retinaface-pytorch -c constraints.txt
#from retinaface.pre_trained_models import get_model

# pip install mtcnn-onnxruntime
# edit nano /home/ubuntu/anaconda3/envs/py36_torch14/lib/python3.6/site-packages/mtcnn_ort/onnx_runner.py
# from cv2 import cv2 -> import cv2
from mtcnn_ort import MTCNN

detector = MTCNN()

def rot90(v):
    return np.array([-v[1], v[0]])

class CreateCeleba:
    def __init__(self, infolder):

        self.size = 256
        
        #self.detect_model = get_model("resnet50_2020-07-20", max_size=2048)
        #self.detect_model.eval()
        self.detect_model = MTCNN()
        
            
        types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
        self.files_grabbed = []
        for files in types:
            self.files_grabbed.extend(glob.glob(infolder + "/**/" + files, recursive=True))
            
            
        
        self.image_filename = self.files_grabbed
        
        self.images = []
        self.image_filename2 = []
        for f in tqdm(self.image_filename):
            image = self.process_func(f)
            if image is not None:
                self.images.append(image)
                self.image_filename2.append(f)
        self.image_filename =self.image_filename2
    

    def process_func(self, orig_file):
        # Load original image.
        img = PIL.Image.open(orig_file)
        
        faces = self.detect_model.detect_faces(np.array(img)) #predict_jsons(np.array(img))
    
        if len(faces) == 0:
            print("No face found")
            return None
    
        #ldmk = np.array(faces[0]["landmarks"]).astype(np.int32)
        #lm = ldmk.reshape((5,2))
        ldmk = faces[0]["keypoints"]
        ldmk = np.stack([ldmk['left_eye'],ldmk['right_eye'],ldmk['nose'], ldmk['mouth_left'], ldmk['mouth_right']]).astype(np.int32)#, axis=0 ).astype(np.int32)
        lm = ldmk.reshape((5,2))

        # Choose oriented crop rectangle.
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = self.size / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(self.size * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(self.size * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((4*self.size, 4*self.size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((self.size, self.size), PIL.Image.ANTIALIAS)
        
        return img




class Faces(Dataset):
    """docstring for _3DMM"""
    def __init__(self, args, celeba, root, transform, split='train'):
        super(Faces, self).__init__()
        self.root = root
        self.transform = transform
        self.totensor = torchvision.transforms.ToTensor()
        
        self.celeba = celeba

        self.image_filename = self.celeba.files_grabbed
        
        self.images = self.celeba.images
        self.image_filename =self.celeba.image_filename
            

        self.train_percent = args.train_dev_percent
        self.split = split
        if split == 'train':
            self.offset = 0
            self.len = int(len(self.images) * self.train_percent)
        elif split == 'test':
            self.offset = int(len(self.images) * self.train_percent)
            self.len = int(len(self.images) * (1 - self.train_percent))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx += self.offset
        
        image = self.images[idx]
        
        if self.transform:
            if hasattr(self.transform, 'get_class_fullname') and 'albumentations' in self.transform.get_class_fullname():
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        return image, self.image_filename[idx]






class Dataloader:
    def __init__(self, args):
        self.args = args
        self.dist = args.dist
        if self.dist:
            self.world_size = args.ngpu
            self.rank = args.rank

        self.dataset_options = args.dataset_options
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train

        self.train_dev_percent = args.train_dev_percent
        self.test_dev_percent = args.test_dev_percent
        self.resolution = args.resolution

        celeba = CreateCeleba(self.args.dataroot)

        if self.dataset_train_name == 'Faces':
            self.dataset_train = Faces(
                args=args,
                celeba=celeba,
                root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='train'
            )

        elif hasattr(datasets, self.dataset_train_name):
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                args=self.args,
                root=self.args.dataroot,# + "/300W_LP",
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='train'
            )


        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'Faces':
            self.dataset_test = Faces(
                args=args,
                celeba=celeba,
                root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='test'
            )

        elif hasattr(datasets, self.dataset_test_name):
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                args=args,
                root=self.args.dataroot,# + "/300W_LP",
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='test'
            )

        else:
            # raise(Exception("Unknown Dataset"))
            return

    def create(self, flag=None, shuffle=True):
        dataloader = {}

        train_sampler = DistributedSampler(self.dataset_train, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle) if self.dist else None
        test_sampler = DistributedSampler(self.dataset_test, num_replicas=self.world_size, rank=self.rank, shuffle=False) if self.dist else None

        if flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(train_sampler is None and shuffle), pin_memory=True,
                sampler = train_sampler
            )
            dataloader['train_sampler'] = train_sampler

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True,
                sampler = test_sampler
            )
            dataloader['test_sampler'] = test_sampler

            dataloader['eval'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(test_sampler is None and shuffle), pin_memory=True,
                sampler = test_sampler
            )
            return dataloader

        elif flag.lower() == "train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(train_sampler is None and shuffle), pin_memory=True,
                sampler = train_sampler
            )
            dataloader['train_sampler'] = train_sampler
            return dataloader

        elif flag.lower() == "test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True,
                sampler = test_sampler
            )
            dataloader['test_sampler'] = test_sampler
            return dataloader

    def create_subsets(self, flag=None, shuffle=False):
        dataloader = {}

        if flag is None:
            train_len = len(self.dataset_train)
            train_cut_index = int(train_len * (self.train_dev_percent))
            indices = list(torch.arange(train_len))
            train_indices = indices[:train_cut_index]
            test_indices = indices[train_cut_index:]

            train_sampler = torch.utils.data.Subset(self.dataset_train, train_indices)
            test_sampler = torch.utils.data.Subset(self.dataset_train, test_indices)

            dataloader['train'] = torch.utils.data.DataLoader(
                train_sampler, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                test_sampler, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['eval'] = torch.utils.data.DataLoader(
                test_sampler, batch_size=self.args.batch_size,
                shuffle=True, num_workers=self.args.nthreads,
                pin_memory=True
            )

        elif flag.lower() == 'train':
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=self.args.nthreads,
                pin_memory=True
            )
        elif flag == "all":
            dataloader['all'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.nthreads,
                pin_memory=True
            )

        return dataloader





def main():
    # parse the arguments
    args = config.parse_args()
    '''
    if (args.ngpu > 0 and torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"
    '''
    device = "cpu" #"cuda:0"
    args.device = torch.device(device)
    args.rank = 0
    
    args.eval = True

    # set my data root
    args.dataroot =  'inputs'


    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if args.save_results:
        utils.saveargs(args)

    print("Create model")
    # Create Model
    models = Model(args)
    model, criterion = models.setup(split_devices=False)

    print("Create occlusion segmentation")
    # Occlusion segmentation model
    modelO = None
    modelO = UnetSeg(args, in_channels=3, out_channels=3)
    modelO.to(args.device)
    if args.ngpu > 1:
        modelO = nn.DataParallel(modelO, list(range(args.ngpu)))

    print("create dataloader")
    # Data Loading
    dataloader = Dataloader(args)

    # loaders = dataloader.create(shuffle=True)
    loaders = dataloader.create_subsets(shuffle=True)       # Only for CelebA

    print("create trainer/tester")
    # Initialize trainer and tester
    trainer = Trainer(args, model, modelO, criterion, args.device)
    tester = Tester(args, model, modelO, trainer.modelR, criterion, trainer.renderer, tb_writer=None)

    if args.eval:
        from eval import Evaluator
        evaluator = Evaluator(args, model, modelO, trainer.modelR, trainer.renderer, device=args.device, tb_writer=None)
        loaders = dataloader.create(shuffle=False)
        print("0")
        with torch.no_grad():
            evaluator.evaluate(0, loaders['train']) # change to test for celeba
        return

    # Run training/testing
    if args.test:
        loss_test = tester.test(0, test_loader)
        tester.tb_writer.close()


if __name__ == "__main__":
    main()
