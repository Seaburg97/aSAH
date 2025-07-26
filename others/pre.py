# --coding:utf-8--
import torch
import torch.nn as nn
import torchio as tio
import numpy as np

class RandomContrast(tio.Transform):
    def __init__(self, augmentation_factor=(0.75, 1.25), **kwargs):
        super().__init__(**kwargs)
        self.augmentation_factor = augmentation_factor

    def apply_transform(self, subject):
        for image in subject.get_images_dict().values():
            contrast_factor = np.random.uniform(self.augmentation_factor[0], self.augmentation_factor[1])
            array = image.numpy()
            mean = array.mean()
            array = (array - mean) * contrast_factor + mean
            image.set_data(torch.tensor(array))
        return subject


class RandomResizedCrop3D(tio.Transform):
    def __init__(self, output_size, scale, ratio, p=1.0):
        super().__init__(p=p)
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio

    def apply_transform(self, subject):

        image = subject['image']
        depth, height, width = image.spatial_shape

   
        scale_factor = torch.empty(1).uniform_(*self.scale).item()
        crop_size = tuple(int(s * scale_factor) for s in (depth, height, width))


        crop_transform = tio.RandomCrop(crop_size, p=1.0)
        cropped_subject = crop_transform(subject)


        resize_transform = tio.Resize(self.output_size)
        resized_subject = resize_transform(cropped_subject)

        return resized_subject