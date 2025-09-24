import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# 加载数据
final_data = pd.read_csv('final_data.csv')

# 将传感器数据和标签分开
X = final_data.iloc[:, :-1].values  # 所有列除了标签列
y = final_data['label'].values  # 标签列

# 将X重塑为适合TimeSformer的输入格式 (batch_size, 120, 3)
X = X.reshape(-1, 120, 3)  # (样本数, 120, 3)

# 将数据分割为训练集、验证集和测试集 (80% 训练，10% 验证，10% 测试)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 PyTorch DataLoader（批量加载数据）
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 定义训练集和验证集的 DataLoader
batch_size = 32  # 可以根据需要调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型（假设你已经定义了 TimeSformer 模型）
model = TimeSformer(num_classes=3, patch_size=120, num_channels=3, dim=256, num_heads=8, num_layers=6, dropout_rate=0.1)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 模型训练
num_epochs = 10  # 可以根据需要调整
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for data, labels in train_loader:
        data = data.float()  # 确保数据类型为浮动类型
        labels = labels.long()  # 确保标签是整数类型

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.float()
            labels = labels.long()

            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy*100:.2f}%")

# 在测试集上进行评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        data = data.float()
        labels = labels.long()

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
