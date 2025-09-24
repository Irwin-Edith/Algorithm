import torch
import torch.nn as nn
import torch.nn.functional as F

# TimeSformer模型类
class TimeSformer(nn.Module):
    def __init__(self, num_classes=3, patch_size=120, num_channels=3, dim=256, num_heads=8, num_layers=6, dropout_rate=0.1):
        """
        初始化TimeSformer模型
        Args:
            num_classes: 类别数（默认为3，表示三种表面类型）
            patch_size: 每个patch的大小（即测点数量，默认为120）
            num_channels: 输入数据的通道数（默认为3，表示每个测点的xyz分力）
            dim: Transformer的维度（隐藏层维度）
            num_heads: 自注意力头的数量
            num_layers: Transformer编码器层数
            dropout_rate: Dropout的概率
        """
        super(TimeSformer, self).__init__()
        
        # 1. 输入嵌入层：将每个测点的xyz力值（3个方向）嵌入为一个高维向量
        self.embedding = nn.Linear(patch_size * num_channels, dim)  # (120 * 3) -> dim

        # 2. 位置嵌入：为每个测点的位置添加位置编码
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_size, dim))
        
        # 3. Transformer编码器层
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )

        # 4. 分类头：用于输出分类结果
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入数据，形状为 (batch_size, patch_size, num_channels)，即 (batch_size, 120, 3)
        """
        # 1. 将每个测点的 xyz 力值嵌入为一个高维向量
        x = self.embedding(x.view(x.size(0), -1))  # 变为 (batch_size, dim)

        # 2. 加入位置嵌入
        x = x + self.position_embedding

        # 3. 经过Transformer编码器层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # 4. 进行分类
        x = self.fc(x.mean(dim=1))  # 对所有测点的输出求均值作为分类输入

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate):
        """
        Transformer 编码器层
        Args:
            dim: Transformer的维度（隐藏层维度）
            num_heads: 自注意力头的数量
            dropout_rate: Dropout的概率
        """
        super(TransformerBlock, self).__init__()

        # 自注意力层
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入数据，形状为 (batch_size, seq_len, dim)
        """
        # 自注意力层
        attn_output, _ = self.attention(x, x, x)  # (batch_size, seq_len, dim)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 层归一化

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # 残差连接 + 层归一化

        return x

