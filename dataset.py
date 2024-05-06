import glob
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from renderer.camera import compute_cam2world_matrix, sample_rays
from renderer.utils import TensorGroup

flip = T.RandomHorizontalFlip()

class Imagenet_Dataset(Dataset):
    def __init__(self, folder, image_size=64, config=None, random_flip=True, get_224=False):
        super(Imagenet_Dataset).__init__()
        self.random_flip = random_flip
        if not config.dataset_params.all_classes:
            self.image_names = glob.glob(folder + "*.jpg")#[:16]
        else:
            # get the paths for the image files
            # can be further improved
            self.image_names = []
            folder_names = os.listdir(folder)
            for f in folder_names:
                self.image_names.extend(glob.glob(os.path.join(folder, f) + "/*.jpg"))
        print('Number of files: ', len(self.image_names))
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
            np.array,
            T.ToTensor()
        ])
        self.get_224 = get_224
        if get_224:
            output_size = 224
            print('**Getting Size: ', output_size)
            self.transform_224 = T.Compose([
                T.Resize(output_size, interpolation=T.InterpolationMode.LANCZOS),
                np.array,
                T.ToTensor()
            ])

        self.config = config

    def __len__(self):
        return len(self.image_names)

    def get_names(self):
        return self.image_names

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]

        depth_filename = img_filename.replace('.jpg', '_depth.png')

        results = {}
        results['idx'] = idx
        results['name'] = img_filename.split('/')[-1]
        images = Image.open(img_filename)
        results['depth'] = Image.open(depth_filename)
        results['images'] = self.transform(images)
        results['depth'] = self.transform(results['depth'])

        if results['depth'].shape[0] > 1:
            results['depth'] = results['depth'][0].unsqueeze(0)

        results['depth'] = results['depth'] / 65536 * 2.0 + 4
        if self.get_224:
            results['images_224'] = self.transform_224(images)

        if self.random_flip:
            if random.random() < 0.5:
                results['idx'] = idx + len(self.image_names)
                results['images'] = flip(results['images'])
                results['depth'] = flip(results['depth'])
                if self.get_224:
                    results['images_224'] = flip(results['images_224'])

        return results


class Custom_Dataset(Dataset):
    def __init__(self, folder, image_size,  config=None):
        super(Imagenet_Dataset).__init__()
        image_names_jpg = glob.glob(folder + "/*/*.jpg")
        image_names_png = glob.glob(folder + "/*/*.png")
        image_names = image_names_jpg + image_names_png
        self.image_names = [image_orginal for image_orginal in image_names if 'depth' not in image_orginal]

        print('Number of files: ', len(self.image_names) // 2)

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            np.array,
            T.ToTensor()
        ])

        self.config = config

    def __len__(self):
        return len(self.image_names)

    def get_names(self):
        return self.image_names

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]

        depth_filename_png = img_filename.replace('.jpg', '_depth.png')
        depth_filename_jpg = img_filename.replace('.jpg', '_depth.jpg')

        depth_filename = depth_filename_png if os.path.exists(depth_filename_png) else depth_filename_jpg

        results = {}
        results['idx'] = idx
        results['name'] = img_filename.split('/')[-1]
        image = Image.open(img_filename)
        results['depth'] = Image.open(depth_filename)
        results['images'] = self.transform(image)
        results['depth'] = self.transform(results['depth'])

        if results['depth'].shape[0] > 1:
            results['depth'] = results['depth'][0].unsqueeze(0)

        results['depth'] = results['depth'] / 65536 * 2.0 + 4

        if self.config.randomFLip and random.random() < 0.5:
            results['idx'] = idx + len(self.image_names)
            results['images'] = flip(results['images'])
            results['depth'] = flip(results['depth'])

        return results

def getDataset(name):
    if name == 'Imagenet':
        return Imagenet_Dataset
    elif name == 'Custom':
        return Custom_Dataset
    else:
        return Dataset


def get_camera_rays(image_size):
    fov = 18
    radius = 5
    fov_degrees = torch.ones(1, 1) * fov
    camera_params = TensorGroup(
        angles=torch.zeros(1, 3),
        # fov=0,
        radius=torch.ones(1, 1) * radius,
        look_at=torch.zeros(1, 3),
    )

    camera_params.angles[:, 0] = np.pi / 2  # +np.pi/18
    camera_params.angles[:, 1] = np.pi / 2  # +np.pi/18

    cam2w = compute_cam2world_matrix(camera_params)
    ray_origins, ray_directions = sample_rays(cam2w, fov_degrees, (image_size, image_size))
    rays_o = ray_origins.reshape(-1, 3)
    rays_d = ray_directions.reshape(-1, 3)

    return rays_o, rays_d