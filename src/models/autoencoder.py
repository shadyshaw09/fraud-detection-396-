"""
Autoencoder Model for Unsupervised Anomaly Detection

Uses a deep neural network autoencoder trained on normal transactions.
The model learns to reconstruct normal data patterns.

Theory:
- Train on majority of data (assumed normal)
- Model learns the manifold of normal transactions
- Anomalies have higher reconstruction error
- Reconstruction error = anomaly score
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class AutoencoderNetwork(nn.Module):
    """
    Deep Autoencoder neural network architecture.
    
    Encoder compresses features to bottleneck,
    Decoder reconstructs from bottleneck.
    Reconstruction error indicates anomalies.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 4):
        """
        Initialize autoencoder architecture.
        
        Args:
            input_dim: Number of input features (dimension of risk embedding)
            latent_dim: Dimension of bottleneck layer (compressed representation)
        """
        super().__init__()
        
        # Encoder: compress input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        
        # Decoder: expand latent_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )
    
    def encode(self, x):
        """Compress to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Reconstruct from latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode then decode."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class AutoencoderModel:
    """
    Autoencoder-based anomaly detection model.
    
    Learns the distribution of normal transactions through reconstruction,
    then uses reconstruction error as anomaly score.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize Autoencoder model.
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of bottleneck
            learning_rate: Optimizer learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: 'cpu' or 'cuda'
            verbose: Print training progress
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        
        # Choose device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("[AUTOENCODER] Using CUDA (GPU)")
        else:
            self.device = torch.device('cpu')
            print("[AUTOENCODER] Using CPU")
        
        # Initialize network
        self.network = AutoencoderNetwork(input_dim, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.is_fitted = False
        self.training_losses: List[float] = []
        self.reconstruction_error_threshold = None
    
    def fit(self, X: np.ndarray, validation_split: float = 0.1) -> None:
        """
        Fit autoencoder on training data (UNSUPERVISED).
        
        Minimizes reconstruction error on normal transactions.
        No labels needed - assumes majority of data are normal.
        
        Args:
            X: Training feature matrix (n_samples, n_features)
            validation_split: Fraction to use for validation (default 0.1)
        """
        print(f"[AUTOENCODER] Training autoencoder on {X.shape[0]} samples...")
        print(f"[AUTOENCODER] Input dimension: {X.shape[1]}, Latent dimension: {self.latent_dim}")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Split into train and validation
        n_train = int(len(X) * (1 - validation_split))
        X_train = X_tensor[:n_train]
        X_val = X_tensor[n_train:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.network.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstruction, _ = self.network(X_batch)
                
                # Compute loss
                loss = self.criterion(reconstruction, X_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * X_batch.shape[0]
            
            epoch_loss /= len(X_train)
            self.training_losses.append(epoch_loss)
            
            # Validation
            self.network.eval()
            with torch.no_grad():
                val_reconstruction, _ = self.network(X_val)
                val_loss = self.criterion(val_reconstruction, X_val).item()
            self.network.train()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"[AUTOENCODER] Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Compute reconstruction error threshold
        self.network.eval()
        with torch.no_grad():
            all_reconstruction, _ = self.network(X_tensor)
            errors = torch.mean((X_tensor - all_reconstruction) ** 2, dim=1).cpu().numpy()
        
        # Set threshold at 95th percentile of training errors
        self.reconstruction_error_threshold = np.percentile(errors, 95)
        
        self.is_fitted = True
        print(f"[AUTOENCODER] Training complete. Reconstruction error threshold: {self.reconstruction_error_threshold:.6f}")
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for samples.
        
        Higher error = more anomalous.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            reconstruction, _ = self.network(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1).cpu().numpy()
        
        return errors
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized anomaly scores (0-1).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of anomaly scores (0-1, higher = more anomalous)
        """
        errors = self.get_reconstruction_errors(X)
        
        # Normalize to 0-1 range
        error_min = errors.min()
        error_max = errors.max()
        
        if error_max > error_min:
            scores = (errors - error_min) / (error_max - error_min)
        else:
            scores = np.zeros_like(errors)
        
        return scores
    
    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Binary anomaly predictions based on threshold.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predictions (1 for anomaly, -1 for normal)
        """
        if self.reconstruction_error_threshold is None:
            raise ValueError("Reconstruction error threshold not set. Call fit() first.")
        
        errors = self.get_reconstruction_errors(X)
        predictions = np.where(errors > self.reconstruction_error_threshold, 1, -1)
        
        return predictions
    
    def predict_and_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both binary predictions and continuous anomaly scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, scores)
        """
        predictions = self.predict_anomalies(X)
        scores = self.get_anomaly_scores(X)
        
        return predictions, scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (.pt or .pth)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save unfitted model.")
        
        checkpoint = {
            'model_state': self.network.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'threshold': self.reconstruction_error_threshold,
        }
        
        torch.save(checkpoint, filepath)
        print(f"[AUTOENCODER] Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.network = AutoencoderNetwork(self.input_dim, self.latent_dim).to(self.device)
        self.network.load_state_dict(checkpoint['model_state'])
        self.reconstruction_error_threshold = checkpoint['threshold']
        
        self.is_fitted = True
        print(f"[AUTOENCODER] Model loaded from {filepath}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model parameters and statistics
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "status": "Fitted",
            "algorithm": "Autoencoder",
            "input_dimension": self.input_dim,
            "latent_dimension": self.latent_dim,
            "epochs_trained": self.epochs,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "reconstruction_error_threshold": float(self.reconstruction_error_threshold),
            "final_training_loss": float(self.training_losses[-1]) if self.training_losses else None,
        }
