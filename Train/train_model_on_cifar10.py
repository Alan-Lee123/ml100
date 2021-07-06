import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize

from CNN.vgg16 import VGG16
from CNN.resnet18 import Resnet18
from Utils.utils import TrainingProgress

transform_train = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

# Batch Size
batch_size = 512

trainset = torchvision.datasets.CIFAR10(
    root='/data/torchdataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='/data/torchdataset', train=False, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning Rate
lr = 1e-2

# Model
# model = VGG16(10).to(device)
model = Resnet18(10).to(device)

criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def eval():
    model.eval()
    tot = 0
    correct_cnt = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct_cnt += predicted.eq(targets).sum().item()
            tot += len(inputs)
    return correct_cnt / tot



# Training
def train(epochs):
    total_steps = len(trainloader)
    progress = TrainingProgress(epochs, total_steps)

    accumulated_loss = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            _, predicted = outputs.max(1)
            correct_rate = predicted.eq(targets).sum().item() / batch_size

            if batch_idx % 10 == 0:
                progress.show_progress(epoch, batch_idx + 1, train_loss, 
                    optimizer.param_groups[0]['lr'],correct_rate)
            
            accumulated_loss += train_loss
        
        print("Start to evaluate")
        eval_acc = eval()
        print(f'Epoch {epoch} Finished. Test Acc is : {eval_acc}')
        scheduler.step()
    torch.save(model.state_dict(), f'Checkpoints/cifar10.pth')

if __name__=='__main__':
    train(1)

        





