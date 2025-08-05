"""dimension reduction"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """feature processor, provide normalization and dimension reduction"""
    
    def __init__(self, method: str = "none", n_components: int = None):
        """
        initialize feature processor
        
        Args:
            method: processing method, optional: "none", "pca", "autoencoder"
            n_components: dimension after reduction, None means automatic decision
        """
        self.method = method
        self.n_components = n_components
        self.processor = None
        self.is_fitted = False
        self.input_dim = None
        self.output_dim = None
        
    def fit(self, features: np.ndarray) -> None:
        """
        train feature processor
        
        Args:
            features: feature matrix, shape: [n_samples, n_features]
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
                
                # determine the number of components
                if self.n_components is None:
                    # automatically select the number of components that can explain 95% of the variance
                    self.n_components = min(features.shape[0], features.shape[1])
                    
                self.processor = PCA(n_components=self.n_components)
                self.processor.fit(features)
                
                # calculate the actual output dimension
                explained_variance_ratio = self.processor.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.output_dim = np.sum(cumulative_variance <= 0.95) + 1
                self.output_dim = min(self.output_dim, self.input_dim)
                
                self.is_fitted = True
                logger.info(f"PCA fit successfully, input dimension: {self.input_dim}, output dimension: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"PCA fit failed: {e}")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
                
        elif self.method == "autoencoder":
            try:
                # autoencoder implementation
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
                
                # determine the hidden layer dimension
                if self.n_components is None:
                    self.n_components = max(1, self.input_dim // 2)
                
                # create and train autoencoder
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.processor = Autoencoder(self.input_dim, self.n_components).to(device)
                features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.processor.parameters(), lr=0.001)
                
                # training
                for epoch in range(100):
                    # forward propagation
                    outputs = self.processor(features_tensor)
                    loss = criterion(outputs, features_tensor)
                    
                    # backward propagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                self.output_dim = self.n_components
                self.is_fitted = True
                logger.info(f"autoencoder fit successfully, input dimension: {self.input_dim}, output dimension: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"autoencoder fit failed: {e}")
                logger.warning("fallback to no dimension reduction")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
        else:
            logger.warning(f"unknown dimension reduction method: {self.method}, will not perform dimension reduction")
            self.method = "none"
            self.output_dim = self.input_dim
            self.is_fitted = True
            
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        transform features
        
        Args:
            features: input features, shape: [n_samples, n_features] or [n_features]
            
        Returns:
            transformed features
        """
        if not self.is_fitted:
            logger.warning("processor not fitted, return original features")
            return features
            
        single_sample = False
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            single_sample = True
            
        # ensure the input dimension is correct
        if features.shape[1] != self.input_dim:
            logger.error(f"input dimension mismatch, expected {self.input_dim}, actual {features.shape[1]}")
            # try to fix by zero padding or truncation
            if features.shape[1] < self.input_dim:
                # zero padding
                padded = np.zeros((features.shape[0], self.input_dim))
                padded[:, :features.shape[1]] = features
                features = padded
            else:
                # truncation
                features = features[:, :self.input_dim]
                
        if self.method == "none":
            result = features
        elif self.method == "pca":
            try:
                result = self.processor.transform(features)
                # only keep the components that can explain 95% of the variance
                result = result[:, :self.output_dim]
            except Exception as e:
                logger.error(f"PCA transform failed: {e}")
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
                logger.error(f"autoencoder transform failed: {e}")
                result = features
                
        if single_sample:
            result = result.reshape(-1)
            
        return result
    
    def get_output_dim(self) -> int:
        """get output dimension"""
        if not self.is_fitted:
            raise ValueError("processor not fitted, cannot get output dimension")
        return self.output_dim 