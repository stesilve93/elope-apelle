
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from config import configs
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1

class EventData(Dataset):
    
    """
    args:
    data_folder_path:the path of data
    split:'train' or 'test'
    """
    
    def __init__(
        self, 
        data_folder_path: str, 
        split: str, 
        count_only: bool = False, 
        time_only: bool = False, 
        skip_frames: bool = False
    ):
        
        # Store internal settings
        self._data_folder_path = data_folder_path
        self._split = split
        self._count_only = count_only
        self._time_only = time_only
        self._skip_frames = skip_frames
        
        self.args = configs()
        
        self.event_data_paths, self.n_ima = self.read_file_paths(
            self._data_folder_path, self._split
        )

    def __getitem__(self, index: int) -> tuple:
        # image_times event_count_images event_time_images image_iter prefix cam
        
        image_iter = 0
        for i in self.n_ima:
            if index < i:
                break
            image_iter += 1
        
        image_iter -= 1
        cam = 'left' if image_iter % 2 == 0 else 'right'
            
        prefix = self.event_data_paths[image_iter]
        image_iter = index - self.n_ima[image_iter]

        # Load the data file
        img_path = prefix + "/" + cam + "_event" + str(image_iter).rjust(5, '0') + ".npy"
        event_count_images, event_time_images, image_times = np.load(
            img_path, encoding="bytes", allow_pickle=True
        )
        
        # Convert to PyTorch tensors 
        event_count_images = torch.from_numpy(event_count_images.astype(np.int16))
        event_time_images  = torch.from_numpy(event_time_images.astype(np.float32))
        image_times        = torch.from_numpy(image_times.astype(np.float64))

        # Check how many frames to aggregate
        if self._split is 'test':
            n_frames = _TEST_SKIP_FRAMES if self._skip_frames else 1
        else:
            n_frames = np.random.randint(low=1, high=_MAX_SKIP_FRAMES+1) * _N_SKIP
            
        timestamps = [image_times[0], image_times[n_frames]]
        img_count, img_stamp = self._read_events(event_count_images, event_time_images, n_frames)

        img1_path = prefix + "/" + cam + "_image" + str(image_iter).rjust(5,'0') + ".png"
        img2_path = prefix + "/" + cam + "_image" + str(image_iter+n_frames).rjust(5,'0') + ".png"

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Dataset augmentations
        rand_flip = np.random.randint(low=0, high=2)
        rand_rotate = np.random.randint(low=-30, high=30)
        
        # Retrieve image shapes 
        H, W = self.args.image_height, self.args.image_width
        
        x = np.random.randint(low=1, high=(img_count.shape[1] - H))
        y = np.random.randint(low=1, high=(img_count.shape[2] - W))
        
        if self._split == 'train':
            
            if self._count_only:
                
                # Transform image to 0-1 (this is not normalized!, it contains the total 
                # number of events that happened)
                img_count = F.to_pil_image(img_count / 255.)
                
                # Apply random flip
                if rand_flip == 0:
                    img_count = img_count.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # Apply random rotation
                event_image = img_count.rotate(rand_rotate)
                
                # Apply random crop and re-rebring image to actual values
                event_image = F.to_tensor(event_image) * 255.
                event_image = event_image[:, x:x+H, y:y+W]
                
            elif self._time_only:
                
                # Retrieve the image with the latest timestamps (this is normalized)
                img_stamp = F.to_pil_image(img_stamp)
                
                # Apply random flip 
                if rand_flip == 0:
                    img_stamp = img_stamp.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # Apply random rotation
                event_image = img_stamp.rotate(rand_rotate)
                
                # Apply random crop
                event_image = F.to_tensor(event_image)
                event_image = event_image[:, x:x+H, y:y+W]
                
            else:
                
                # Use both the total number of events as well as the last stamp
                img_count = F.to_pil_image(img_count / 255.)
                img_stamp = F.to_pil_image(img_stamp)
                
                # Apply random flip
                if rand_flip == 0:
                    img_count = img_count.transpose(Image.FLIP_LEFT_RIGHT)
                    img_stamp = img_stamp.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # Apply random rotation
                event_count = img_count.rotate(rand_rotate)
                event_stamp = img_stamp.rotate(rand_rotate)
                
                # Convert the images to tensor and un-normalize
                event_count = F.to_tensor(event_count)
                event_stamp = F.to_tensor(event_stamp) * 255.
                
                # Stack the two images together and randomly crop them
                event_image = torch.cat((event_count, event_stamp), dim=0)
                event_image = event_image[..., x:x+H, y:y+W]

            if rand_flip == 0:
                # Check whether the ground-truth should be flipped aswell
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Apply the same random rotation     
            img1 = img1.rotate(rand_rotate)
            img2 = img2.rotate(rand_rotate)
            
            # Convert to tensors
            img1 = F.to_tensor(img1)
            img2 = F.to_tensor(img2)
            
            # Crop the images
            img1 = img1[..., x:x+H, y:y+W]
            img2 = img2[..., x:x+H, y:y+W]
            
            return event_image, img1, img2, timestamps
        
        # If we end-up here, we are in the testing modality
        if self._count_only: 
            
            # Retrieve the image, normalize, crop, and un-normalize
            event_count = F.center_crop(F.to_pil_image(img_count / 255.0), (H, W))
            event_image = F.to_tensor(event_count) * 255.0
            
        elif self._time_only:
            
            # Retrieve the image, crop, and un-normalize
            event_stamp = F.center_crop(F.to_pil_image(img_stamp), (H, W))
            event_image = F.to_tensor(event_stamp)
            
        else:
            
            # Stack the two images together and crop 
            event_image = torch.cat((img_count / 255.0, img_stamp), dim=0)
            event_image = F.center_crop(F.to_pil_image(event_image), (H, W))
            event_image = F.to_tensor(event_image)
            
            # Un-normalize the event counts 
            event_image[:2, ...] = event_image[:2, ...] * 255.0

        # Apply the same cropping to the ground-truth
        img1 = F.to_tensor(F.center_crop(img1, (H, W)))
        img2 = F.to_tensor(F.center_crop(img2, (H, W)))

        return event_image, img1, img2, timestamps

    def __len__(self):
        return self.n_ima[-1]

    def _read_events(
        self,
        images_event_count: torch.Tensor,
        images_event_stamp: torch.Tensor,
        n_frames: int
    ) -> tuple:
        
        # images_event_count (N, H, W, P) with P = 2
        # images_event_stamp (N, H, W, P) with P = 2
        
        # Aggregate the total event count among multiple images
        img_count = images_event_count[:n_frames, :, :, :]
        img_count = torch.sum(img_count, dim=0).type(torch.float32)
     
        img_count = img_count.permute(2,0,1) # (2, H, W)

        img_stamp = images_event_stamp[:n_frames, :, :, :]
        img_stamp = torch.max(img_stamp, dim=0)[0]

        img_stamp /= torch.max(img_stamp)
        img_stamp = img_stamp.permute(2,0,1) # (2, H, W)

        return img_count, img_stamp

    def read_file_paths(
        self,
        data_folder_path: str,
        split: str,
        sequence=None
    ) -> tuple:
        """
        return: event_data_paths,paths of event data (left and right in one folder is two)
        n_ima: the sum number of event pictures in every path and the paths before
        """
        

        if sequence is None:
            bag_list_file = open(
                os.path.join(data_folder_path, "{}_bags.txt".format(split)), 'r'
            )
            
            lines = bag_list_file.read().splitlines()
            bag_list_file.close()
            
        else:
            if isinstance(sequence, (list, )):
                lines = sequence
            else:
                lines = [sequence]
        
        event_data_paths, n_ima = [], [0]
        for line in lines:
            bag_name = line

            event_data_paths.append(os.path.join(data_folder_path, bag_name))
            num_ima_file = open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r')
            num_imas = num_ima_file.read()
            num_ima_file.close()
            num_imas_split = num_imas.split(' ')
            n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
            n_ima.append(n_left_ima + n_ima[-1])
            
            n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
            if n_right_ima > 0 and not split is 'test':
                n_ima.append(n_right_ima + n_ima[-1])
                
            else:
                n_ima.append(n_ima[-1])
                
            event_data_paths.append(os.path.join(data_folder_path, bag_name))

        return event_data_paths, n_ima

# if __name__ == "__main__":
#     data = EventData('/media/cyrilsterling/D/EV-FlowNet-pth/data/mvsec/', 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=True)
#     it = 0
#     for i in EventDataLoader:
#         a = i[0][0].numpy()
#         b = i[1][0].numpy()
#         c = i[2][0].numpy()
#         cv2.namedWindow('a')
#         cv2.namedWindow('b')
#         cv2.namedWindow('c')
#         a = a[2,...]+a[3,...]
#         print(np.max(a))
#         a = (a-np.min(a))/(np.max(a)-np.min(a))
#         b = np.transpose(b,(1,2,0))
#         c = np.transpose(c,(1,2,0))
#         cv2.imshow('a',a)
#         cv2.imshow('b',b)
#         cv2.imshow('c',c)
#         cv2.waitKey(1)