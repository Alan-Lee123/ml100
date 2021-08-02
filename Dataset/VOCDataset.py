import torch
import random
from torchvision.datasets import VOCDetection
from torchvision import transforms

name_to_label = {'bus': 1, 'car': 2, 'pottedplant': 3, 'chair': 4, 'diningtable': 5, 'person': 6, 'bird': 7, 'dog': 8, 'cat': 9, 'motorbike': 10, 'train': 11, 'bottle': 12, 'boat': 13, 'tvmonitor': 14, 'bicycle': 15, 'sheep': 16, 'horse': 17, 'cow': 18, 'sofa': 19, 'aeroplane': 20}

class VOCDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_set, download) -> None:
        super().__init__()
        self.dataset = VOCDetection(root_dir, image_set=image_set, download=download)
        self.image_set = image_set
        self.totensor = transforms.ToTensor()
        self.name_to_label = {}
    
    def __getitem__(self, idx):
        img, raw_target = self.dataset.__getitem__(idx)
        img = self.totensor(img.convert("RGB"))
        target = {}
        boxes = []
        labels = []
        areas = []
        annotations = raw_target['annotation']['object']
        for anno in annotations:
            name = anno['name']
            bndbox = anno['bndbox']
            bbox = [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']]
            bbox = [float(v) for v in bbox]
            label = name_to_label[name]
            boxes.append(bbox)
            labels.append(label)
            areas.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        
        # Random Flip
        # if self.image_set == 'train':
        #     if random.randint(0, 1) == 1:
        #         C, H, W = img.shape
        #         img = torch.flip(img, (2,))
        #         for k in range(len(boxes)):
        #             boxes[k] = [W - boxes[k][2], boxes[k][1], W - boxes[k][0], boxes[k][3]]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = areas

        return img, target
    
    def __len__(self):
        return len(self.dataset)


