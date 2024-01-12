from tensorboardX import SummaryWriter

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from model.ResNet import *

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = ImageFolder(root='data/imgs/GADF', transform=transform)

# 划分数据集为训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 检查数据集类别数
num_classes = len(dataset.classes)
print("类别数:", num_classes)

# 创建TensorboardX的SummaryWriter
writer = SummaryWriter()

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetWithAttention(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print("第{}次".format(i))

        # 每隔一定步数记录训练损失到TensorboardX
        if i % 10 == 0:
            step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), step)
            print("train loss = {}".format(loss.item()))

    average_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}')

    # 记录验证损失和准确率到TensorboardX
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = correct / total
    print("Accuracy = {}".format(val_accuracy))
    writer.add_scalar('Validation/Loss', val_loss / len(val_loader), epoch)
    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)

# 关闭TensorboardX的SummaryWriter
writer.close()
torch.save(model.state_dict(), 'save/model.pth')