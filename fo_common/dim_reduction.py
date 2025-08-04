"""Dimensionality reduction functionality"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# Create logger
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Feature processor, providing normalization and dimensionality reduction functionality"""
    
    def __init__(self, method: str = "none", n_components: int = None):
        """
        Initialize feature processor
        
        Args:
            method: Processing method, options: "none", "pca", "autoencoder"
            n_components: Dimensions after reduction, None means auto-determine
        """
        self.method = method
        self.n_components = n_components
        self.processor = None
        self.is_fitted = False
        self.input_dim = None
        self.output_dim = None
        
    def fit(self, features: np.ndarray) -> None:
        """
        Train feature processor
        
        Args:
            features: Feature matrix, shape [n_samples, n_features]
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
                
                # Determine number of components
                if self.n_components is None:
                    # Automatically select number of components that explain 95% variance
                    self.n_components = min(features.shape[0], features.shape[1])
                    
                self.processor = PCA(n_components=self.n_components)
                self.processor.fit(features)
                
                # Calculate actual output dimension
                explained_variance_ratio = self.processor.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.output_dim = np.sum(cumulative_variance <= 0.95) + 1
                self.output_dim = min(self.output_dim, self.input_dim)
                
                self.is_fitted = True
                logger.info(f"PCA fitting successful, input dimension: {self.input_dim}, output dimension: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"PCA fitting failed: {e}")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
                
        elif self.method == "autoencoder":
            try:
                # Simple autoencoder implementation
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
                
                # Determine hidden layer dimension
                if self.n_components is None:
                    self.n_components = max(1, self.input_dim // 2)
                
                # Create and train autoencoder
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.processor = Autoencoder(self.input_dim, self.n_components).to(device)
                features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.processor.parameters(), lr=0.001)
                
                # Simple training
                for epoch in range(100):
                    # Forward pass
                    outputs = self.processor(features_tensor)
                    loss = criterion(outputs, features_tensor)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                self.output_dim = self.n_components
                self.is_fitted = True
                logger.info(f"Autoencoder fitting successful, input dimension: {self.input_dim}, output dimension: {self.output_dim}")
                
            except Exception as e:
                logger.error(f"Autoencoder fitting failed: {e}")
                logger.warning("Falling back to no dimensionality reduction")
                self.method = "none"
                self.output_dim = self.input_dim
                self.is_fitted = True
        else:
            logger.warning(f"Unknown dimensionality reduction method: {self.method}, will not perform reduction")
            self.method = "none"
            self.output_dim = self.input_dim
            self.is_fitted = True
            
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features
        
        Args:
            features: Input features, shape [n_samples, n_features] or [n_features]
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            logger.warning("Processor not fitted, returning original features")
            return features
            
        single_sample = False
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            single_sample = True
            
        # Ensure input dimension is correct
        if features.shape[1] != self.input_dim:
            logger.error(f"Input dimension mismatch, expected {self.input_dim}, got {features.shape[1]}")
            # Try to fix by zero padding or truncation
            if features.shape[1] < self.input_dim:
                # Zero padding
                padded = np.zeros((features.shape[0], self.input_dim))
                padded[:, :features.shape[1]] = features
                features = padded
            else:
                # Truncation
                features = features[:, :self.input_dim]
                
        if self.method == "none":
            result = features
        elif self.method == "pca":
            try:
                result = self.processor.transform(features)
                # Only keep components explaining 95% variance
                result = result[:, :self.output_dim]
            except Exception as e:
                logger.error(f"PCA transformation failed: {e}")
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
                logger.error(f"Autoencoder transformation failed: {e}")
                result = features
                
        if single_sample:
            result = result.reshape(-1)
            
        return result
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        if not self.is_fitted:
            raise ValueError("Processor not fitted, cannot get output dimension")
        return self.output_dim 