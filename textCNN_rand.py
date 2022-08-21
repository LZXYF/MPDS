import torch
from torch import nn
import torch.nn.functional as F


class MHSA(nn.Module):
  def __init__(self, num_heads, dim):
    super().__init__()
    # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.num_heads = num_heads

  def forward(self, x):
    B, N, C = x.shape
    # 生成转换矩阵并分多头
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    # 点积得到attention score
    attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1)

    # 乘上attention score并输出
    v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
    return v

class textCNNModel(torch.nn.Module):

  def __init__(self, pretraining_model, kernel , num_classes):
    super().__init__()
    
    embed_num = 0
    if pretraining_model == "roberta-base":
        embed_num = 50265  #  roberta: 50265 # bert: 30522
    else:
        embed_num = 30522
    embed_dim = 200 # 150
    kernel_num = kernel  # 50
    Ci = 1
    kernel_sizes = [2, 4]
    class_num = num_classes

    self.embedding = nn.Embedding(embed_num, embed_dim)

    self.convs_list = nn.ModuleList(
      [nn.Conv2d(Ci, kernel_num, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])

    self.dropout = nn.Dropout(0.64)
    self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
    self.MHSA = MHSA(10,kernel_num )
    self.bn = torch.nn.BatchNorm1d(len(kernel_sizes) * kernel_num, affine=True, eps=1e-07, momentum=0.005)
    self.init_weight()


  # 初始化权重的函数
  def init_weight(self):
      init_range = 0.2
      self.fc.weight.data.uniform_(-init_range, init_range)
      self.fc.bias.data.zero_()

      init_range = 0.2
      self.embedding.weight.data.uniform_(-init_range, init_range)

  def forward(self, x):
    x = self.embedding(x)
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_list]
    x = [F.max_pool1d(i, i.size(2)).permute(0, 2, 1) for i in x]
    x = torch.cat(x, 1)
    if self.training:
        x = self.dropout(x)

    o = x.view(x.size(0), -1)
    x = self.MHSA(x)
    logit = x.view(x.size(0), -1)
    
    logit = logit + o

    logit = self.fc(logit)
    return logit


