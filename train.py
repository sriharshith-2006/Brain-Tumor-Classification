import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
IMG_SIZE=224
BATCH_SIZE=32
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])
train_dataset=datasets.ImageFolder(
    root='dataset/Train/Train',
    transform=transform
)
test_dataset=datasets.ImageFolder(
    root='dataset/test/test',
    transform=transform
)
train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
img,labels=next(iter(train_loader))
print("Information about the dataset....\n\n")

print(f"Classes and their labels of the dataset are:  {train_dataset.class_to_idx}\n")
print(f"Number of sample of training dataset:  {len(train_dataset)}\n")
print(f"Number of sample of test dataset:  {len(test_dataset)}\n")
print(f"Shape of the Images batch:  {img.shape}\n")
print(f"Shape of labels batch:  {labels.shape}\n")
print(f"Labels of the 1st batch:  {labels}")


