"""维度降低功能"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特征处理器，提供归一化和降维功能"""
    
    def __init__(self, method: str = "none", n_components: int = None):
        """
        初始化特征处理器
        
        Args:
            method: 处理方法，可选："none", "pca", "autoencoder"
            n_components: 降维后的维度，None表示自动决定
        """
        self.method = method
        self.n_components = n_components
        self.processor = None
        self.is_fitted = False
        self.input_dim = None
        self.output_dim = None
        
    def fit(self, features: np.ndarray) -> None:
        """
        训练特征处理器
        
        Args:
            features: 特征矩阵，形状为 [n_samples, n_features]
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        self.input_dim = features.shape[1]
        
        if self.method == "none":
            self.is_fitted = True
            self.output_dim = self.input_dim
            return
            
        elif self.method == "pca":
            try:
                from sklearn.decomposition import PCA
                
                # 确定组件数量
                if self.n_components is None:
                    # 自动选择能够解释95%方差的组件数量
                    self.n_components = min(features.shape[0], features.shape[1])
                    
                self.processor = PCA(n_components=self.n_components)
                self.processor.fit(features)
                
                # 计算实际输出维度
                explained_variance_ratio = self.processor.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.output_dim = np.sum(cumulative_variance <= 0.95) + 1
                self.output_dim = min(self.output_dim, self.input_dim)
                
                self.is_fitted = True
                logger.info(f"PCA拟合成功，输入维度: {self.input_dim}, 输出维度: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"PCA拟合失败: {e}")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
                
        elif self.method == "autoencoder":
            try:
                # 简单自编码器实现
                import torch
                import torch.nn as nn
                import torch.optim as optim
                
                class Autoencoder(nn.Module):
                    def __init__(self, input_dim, hidden_dim):
                        super(Autoencoder, self).__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU()
                        )
                        self.decoder = nn.Sequential(
                            nn.Linear(hidden_dim, input_dim),
                            nn.Sigmoid()
                        )
                        
                    def forward(self, x):
                        encoded = self.encoder(x)
                        decoded = self.decoder(encoded)
                        return decoded
                    
                    def encode(self, x):
                        return self.encoder(x)
                
                # 确定隐藏层维度
                if self.n_components is None:
                    self.n_components = max(1, self.input_dim // 2)
                
                # 创建和训练自编码器
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.processor = Autoencoder(self.input_dim, self.n_components).to(device)
                features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.processor.parameters(), lr=0.001)
                
                # 简单训练
                for epoch in range(100):
                    # 前向传播
                    outputs = self.processor(features_tensor)
                    loss = criterion(outputs, features_tensor)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                self.output_dim = self.n_components
                self.is_fitted = True
                logger.info(f"自编码器拟合成功，输入维度: {self.input_dim}, 输出维度: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"自编码器拟合失败: {e}")
                logger.warning("回退到不进行降维")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
        else:
            logger.warning(f"未知的降维方法: {self.method}，将不进行降维")
            self.method = "none"
            self.output_dim = self.input_dim
            self.is_fitted = True
            
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        转换特征
        
        Args:
            features: 输入特征，形状为 [n_samples, n_features] 或 [n_features]
            
        Returns:
            转换后的特征
        """
        if not self.is_fitted:
            logger.warning("处理器未拟合，返回原始特征")
            return features
            
        single_sample = False
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            single_sample = True
            
        # 确保输入维度正确
        if features.shape[1] != self.input_dim:
            logger.error(f"输入维度不匹配，期望 {self.input_dim}，实际 {features.shape[1]}")
            # 尝试通过零填充或截断修复
            if features.shape[1] < self.input_dim:
                # 零填充
                padded = np.zeros((features.shape[0], self.input_dim))
                padded[:, :features.shape[1]] = features
                features = padded
            else:
                # 截断
                features = features[:, :self.input_dim]
                
        if self.method == "none":
            result = features
        elif self.method == "pca":
            try:
                result = self.processor.transform(features)
                # 只保留解释95%方差的组件
                result = result[:, :self.output_dim]
            except Exception as e:
                logger.error(f"PCA转换失败: {e}")
                result = features
        elif self.method == "autoencoder":
            try:
                import torch
                with torch.no_grad():
                    features_tensor = torch.tensor(features, dtype=torch.float32)
                    if torch.cuda.is_available():
                        features_tensor = features_tensor.cuda()
                    encoded = self.processor.encode(features_tensor)
                    result = encoded.cpu().numpy()
            except Exception as e:
                logger.error(f"自编码器转换失败: {e}")
                result = features
                
        if single_sample:
            result = result.reshape(-1)
            
        return result
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        if not self.is_fitted:
            raise ValueError("处理器未拟合，无法获取输出维度")
        return self.output_dim 