import torch.nn as nn
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,classification_report
from model import CNN_Model
import matplotlib.pyplot as plt
IMG_SIZE=224
BATCH_SIZE=32
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])
test_dataset=datasets.ImageFolder(
    root="dataset/test/test",transform=transform
)
test_loader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)
model=CNN_Model().to(device)
model.load_state_dict(torch.load("saved_models/best_model.pth"))
model.eval()
Loss=nn.CrossEntropyLoss()
test_loss=0
all_pred=[]
all_true=[]
with torch.no_grad():
    for image,label in test_loader:
        img=image.to(device)
        lab=label.to(device)
        outputs=model(img)
        loss=Loss(outputs,lab)
        test_loss+=loss.item()
        _,predicted=torch.max(outputs,dim=1)
        all_pred.extend(predicted.cpu().numpy())
        all_true.extend(lab.cpu().numpy())
    test_loss/=len(test_loader)
    accuracy=accuracy_score(all_true,all_pred)
    recall=recall_score(all_true,all_pred,average='weighted')
    precision=precision_score(all_true,all_pred,average='weighted')
    f1=f1_score(all_true,all_pred,average='weighted')
    cm=confusion_matrix(all_true,all_pred)
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_true,all_pred,target_names=test_dataset.classes))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm,cmap='Blues')
    plt.colorbar()
    classes = test_dataset.classes
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
