import torch
import torch.nn as nn
from torchvision import datasets,transforms
import random
from model import CNN_Model
import matplotlib.pyplot as plt
device='cuda' if torch.cuda.is_available() else 'cpu'
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
predict_dataset=datasets.ImageFolder(
    root="dataset/test/test",
    transform=transform
)
model=CNN_Model().to(device)
model.load_state_dict(
    torch.load("saved_models/best_model.pth", map_location=device)
)
model.eval()
classes=predict_dataset.classes
indices=random.sample(range(len(predict_dataset)),10)
for i,idx in enumerate(indices):
    image,label=predict_dataset[idx]
    input_image=image.unsqueeze(0).to(device)
    output=model(input_image)
    probabilites=torch.softmax(output,dim=1)
    confidence,predictions=torch.max(probabilites,dim=1)
    plt.subplot(2,5,i+1)
    plt.imshow(image.permute(1,2,0))
    plt.title(f"Actual : {classes[label]}\n" f"Pred: {classes[predictions.item()]}\n " f"{confidence.item()*100:2f}%")
    plt.axis("off")
plt.tight_layout()
plt.savefig("predicted images/predictions.png")
plt.show()