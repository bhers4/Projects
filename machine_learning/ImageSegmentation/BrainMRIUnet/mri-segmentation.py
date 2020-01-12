# NTS - Note to Self
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave, imshow
from skimage.transform import resize, rescale, rotate
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pdb

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        self.epochs = 50
        self.batch_size = 16
        self.learning_rate = 0.0001
        self.number_of_threads = 2
        self.features = init_features
        self.encoder1 = self._segmentblock(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._segmentblock(self.features, self.features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._segmentblock(self.features*2, self.features*4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._segmentblock(self.features*4, self.features*8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._segmentblock(self.features*8, self.features*16, name="bottle")
        self.upconv4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._segmentblock((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._segmentblock((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._segmentblock((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = self._segmentblock(self.features * 2, self.features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=self.features, out_channels=out_channels, kernel_size=1
        )

    def _segmentblock(self, in_channels, features, name, kernel_size=3):
        return nn.Sequential(
            OrderedDict([
                (name + "conv1",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False,
                )),
                (name+"batch1",
                 nn.BatchNorm2d(num_features=features)),
                (name+"relu1",
                 nn.ReLU(inplace=True)),
                (name + "conv2",
                 nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False,
                )),
                (name + "batch2",
                 nn.BatchNorm2d(num_features=features)),
                (name + "relu2",
                 nn.ReLU(inplace=True)),
            ])
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

# NTS - CEM
class BrainMRIDataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(
            self,
            images_dir,
            transform=None,
            image_size=256,
            subset="train",
            random_sampling=True,
            seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                    filter(lambda f: ".tif" in f, filenames),
                    key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)
        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=10)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [self.crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [self.pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [self.resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(self.normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def normalize_volume(self, volume):
        p10 = np.percentile(volume, 10)
        p99 = np.percentile(volume, 99)
        volume = rescale_intensity(volume, in_range=(p10, p99))
        m = np.mean(volume, axis=(0, 1, 2))
        s = np.std(volume, axis=(0, 1, 2))
        volume = (volume - m) / s
        return volume

    def resize_sample(self, x, size=256):
        volume, mask = x
        v_shape = volume.shape
        out_shape = (v_shape[0], size, size)
        mask = resize(
            mask,
            output_shape=out_shape,
            order=0,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        out_shape = out_shape + (v_shape[3],)
        volume = resize(
            volume,
            output_shape=out_shape,
            order=2,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        return volume, mask

    def crop_sample(self, x):
        volume, mask = x
        volume[volume < np.max(volume) * 0.1] = 0
        z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
        z_nonzero = np.nonzero(z_projection)
        z_min = np.min(z_nonzero)
        z_max = np.max(z_nonzero) + 1
        y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
        y_nonzero = np.nonzero(y_projection)
        y_min = np.min(y_nonzero)
        y_max = np.max(y_nonzero) + 1
        x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
        x_nonzero = np.nonzero(x_projection)
        x_min = np.min(x_nonzero)
        x_max = np.max(x_nonzero) + 1
        return (
            volume[z_min:z_max, y_min:y_max, x_min:x_max],
            mask[z_min:z_max, y_min:y_max, x_min:x_max],
        )

    def pad_sample(self, x):
        volume, mask = x
        a = volume.shape[1]
        b = volume.shape[2]
        if a == b:
            return volume, mask
        diff = (max(a, b) - min(a, b)) / 2.0
        if a > b:
            padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
        else:
            padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        padding = padding + ((0, 0),)
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
        return volume, mask

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor

batch_size = 16
epochs = 50
lr = 0.0001 # Learning rate
number_of_threads = 2
weights = "./"
image_size = 224
aug_scale = 0.05
aug_angle = 15

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
images_dir = "kaggle_dataset/lgg-mri-segmentation/kaggle_3m"
print("Reading Training Images")
volumes = {}
masks = {}

for (dirpath, dirnames, filenames) in os.walk(images_dir):
    image_slices = []
    mask_slices = []
    for filename in sorted(
            filter(lambda f: ".tif" in f, filenames),
            key=lambda x: int(x.split(".")[-2].split("_")[4]),
    ):
        filepath = os.path.join(dirpath, filename)
        if "mask" in filename:
            mask_slices.append(imread(filepath, as_gray=True))
        else:
            image_slices.append(imread(filepath))
    if len(image_slices) > 0:
        patient_id = dirpath.split("/")[-1]
        volumes[patient_id] = np.array(image_slices[1:-1])
        masks[patient_id] = np.array(mask_slices[1:-1])

test_image = "kaggle_dataset/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif"


