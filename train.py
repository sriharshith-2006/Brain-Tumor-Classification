import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import CNN_Model
import torch.nn as nn
import torch.optim as optim
IMG_SIZE=224
BATCH_SIZE=32
LR=0.0005
EPOCHS=20
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
model=CNN_Model().to(device)
loss=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=LR)
best_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    running_loss=0
    for image,label in train_loader:
        img=image.to(device)
        lab=label.to(device)
        optimizer.zero_grad()
        outputs=model(img)
        Loss=loss(outputs,lab)
        Loss.backward()
        optimizer.step()
        running_loss+=Loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Training Loss: {epoch_loss:.4f}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            model.state_dict(),
            "saved_models/best_model.pth"
        )
        print("Best model saved.")
