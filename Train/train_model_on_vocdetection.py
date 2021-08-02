import torch
import torchvision
import math

from Dataset.VOCDataset import VOCDetectionDataset
from FasterRCNN.FasterRCNN import get_faster_rcnn_model
from Utils.utils import TrainingProgress

batch_size = 1

trainset = VOCDetectionDataset('/data/torchdataset', image_set='train', download=False)
testset = VOCDetectionDataset('/data/torchdataset', image_set='val', download=False)

def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=1,
        collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_faster_rcnn_model(21)
model.to(device)

def train_one_epoch(model, optimizer, data_loader, device, epoch, progress):
    model.train()
    accumulate_loss = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        optimizer.zero_grad()

        images = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        losses.backward()
        optimizer.step()

        accumulate_loss += loss_value

        if (batch_idx + 1) % 100 == 0:
                progress.show_progress(epoch, batch_idx + 1, accumulate_loss / 100, 
                    optimizer.param_groups[0]['lr'], None)
                accumulate_loss = 0
        

def train(num_epochs):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    total_steps = len(data_loader)
    progress = TrainingProgress(num_epochs, total_steps)
    for k in range(num_epochs):
        print(f'start train epoch: {k}')
        train_one_epoch(model, optimizer, data_loader, device, k, progress)
        lr_scheduler.step()