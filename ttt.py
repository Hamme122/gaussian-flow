import torch
from sklearn.neighbors import NearestNeighbors
import numpy
# 假设 origin_vector_part 是你的输入数组，形状为 (37667, 10)
# 用实际数据替换以下随机数据
origin_vector_part = torch.rand(37667, 10)

xyz_coordinates = origin_vector_part[:, :3].cpu().numpy()
K = 10
nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='auto').fit(xyz_coordinates)
distances, indices = nbrs.kneighbors(xyz_coordinates)



# 获取所有最近邻的属性
neighbors = origin_vector_part[indices]  # 形状为 (37667, K, 10)

# 计算所有点与其最近邻之间的属性差异
differences = origin_vector_part.unsqueeze(1) - neighbors  # 形状为 (37667, K, 10)

# 计算差异的二范数并求和
knn_loss = torch.norm(differences, dim=2).sum(dim=1)

# 计算总的 KNN Loss
total_knn_loss = knn_loss.sum()

print(f"Total KNN Loss: {total_knn_loss.item()}")
