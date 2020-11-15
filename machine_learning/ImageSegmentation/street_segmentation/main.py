import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from utils.loader import load_images
from torch.utils.data import Dataset, DataLoader
import random
import json
import cv2 as opencv


class ImageDataset(Dataset):
    def __init__(self, imgs, jsons, load, json_mapping):
        '''
            imgs: list of images
            jsons: list of jsons
        '''
        self.imgs = imgs
        self.jsons = jsons
        self.load = load
        self.xs = []
        self.ys = []
        self.loaded_imgs = {}
        self.loaded_jsons = {}
        # Dict -> Int mapping
        self.dict_int_mapping = {}
        self.json_mapping = json_mapping
        if load:
            for key in imgs.keys():
                # print("Img: ", imgs[key])
                try:
                    img = Image.open(imgs[key])
                    self.loaded_imgs[key] = img
                    self.ys.append(img.size[0])
                    self.xs.append(img.size[1])
                except:
                    print("Error loading Image at ", imgs[key])
            print("Y, min: ", np.min(self.ys), " Max: ", np.max(self.ys), " Mean: ", np.mean(self.ys))
            print("X, min: ", np.min(self.xs), " Max: ", np.max(self.xs), " Mean: ", np.mean(self.xs))
            for key in self.jsons.keys():
                json_mask = self.jsons[key]
                print("Json mask: ", json_mask)
                with open(json_mask, "r") as json_mask:
                    json_mask = json.load(json_mask)
                    img_height = json_mask['imageHeight']
                    img_width = json_mask['imageWidth']
                    img_shapes = json_mask['shapes']
                    img_mask = np.zeros((img_height, img_width))
                    for shape in img_shapes:
                        shape_label = shape['label']
                        polys = shape['points']
                        label_val = self.json_mapping[shape_label]
                        opencv.fillPoly(img_mask, pts=np.int32([polys]), color=label_val)
                    self.loaded_jsons[key] = img_mask
            self.create_dict_to_int_mapping(imgs)
        return

    def create_dict_to_int_mapping(self, data_dict):
        counter = 0
        for key in data_dict:
            self.dict_int_mapping[counter] = key
            self.dict_int_mapping[key] = counter
            counter += 1

    def __len__(self):
        return len(self.imgs.keys())

    def __getitem__(self, i):

        return

def create_json_mapping(jsons):
    label_mapping = {}
    counter = 1
    for key in jsons.keys():
        json_mask = jsons[key]
        print("Json mask: ", json_mask)
        with open(json_mask, "r") as json_mask:
            json_mask = json.load(json_mask)
            # img_height = json_mask['imageHeight']
            # img_width = json_mask['imageWidth']
            img_shapes = json_mask['shapes']
            # img_mask = np.zeros((img_height, img_width))
            for shape in img_shapes:
                shape_label = shape['label']
                polys = shape['points']
                print("Label: ", shape_label, " ", len(polys))
                if shape_label not in label_mapping.keys():
                    label_mapping[shape_label] = counter
                    counter += 1
                label_val = label_mapping[shape_label]
                # opencv.fillPoly(img_mask, pts=np.int32([polys]), color=label_val)
    return label_mapping

# D/datasets
def create_datasets(imgs, jsons, split=0.8, load=True):
    '''
        imgs: Dict
        jsons: Dict
        Split is 0-1 fraction of for example to do 80% training data and 20% testing data
        For very large datasets we might not want to keep all the images in RAM
        Will have to change this to be more generic for RNN/ANN based tasks
    '''
    total_num_imgs = len(imgs.keys())
    total_num_jsons = len(jsons.keys())
    num_train = int(total_num_imgs*split)
    num_test = total_num_imgs - num_train
    print("Total: {},{} Train: {}, Test: {}".format(total_num_imgs, total_num_jsons, num_train, num_test))
    train_keys = random.sample(imgs.keys(), num_train)
    train_imgs = {}
    train_jsons = {}
    test_imgs = {}
    test_jsons = {}
    for key in imgs.keys():
        if key not in train_keys:
            test_jsons[key] = jsons[key]
            test_imgs[key] = imgs[key]
        else:
            train_jsons[key] = jsons[key]
            train_imgs[key] = imgs[key]
    print("Train imgs: {}, Test imgs: {}".format(len(train_imgs), len(test_imgs)))
    # Create coherent json mapping
    json_mapping = create_json_mapping(jsons)
    train_dataset = ImageDataset(train_imgs, train_jsons, load, json_mapping)
    test_dataset = ImageDataset(test_imgs, test_jsons, load, json_mapping)

    return

if __name__ == "__main__":
    img_library = "/home/ben/Documents/img_library"
    imgs, jsons = load_images(img_library)
    create_datasets(imgs, jsons)





'''
    IDEA: one way to load images just for batch is have Image.open( ) in torch dataset.__getitem__(self, i). This is a RAM efficient way
    IDEA: calculate approximate data size, along with RAM target
'''

