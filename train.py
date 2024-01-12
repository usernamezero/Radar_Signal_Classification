from torch.utils.data import DataLoader
from model.CNN import *
# 数据预处理
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# 假设你的数据存放在一个名为 data_dir 的文件夹中，该文件夹下有三个子文件夹分别存放三类图片
data_dir = 'data/imgs/GADF'

# 使用 datasets.ImageFolder 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分数据集为训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 创建模型，并将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNWithSelfAttention(num_classes=3).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 主模块
if __name__ == '__main__':
    # 训练模型
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 在验证集上进行评估
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {(100 * correct / total):.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'save/model.pth')
