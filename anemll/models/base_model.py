from abc import abstractmethod
import torch
import torch.nn as nn
import os
import gc
from tqdm import tqdm
import safetensors.torch

class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, config):
        """Initialize the base model.
        
        Args:
            config: Model configuration object
        """
        super().__init__()  # Initialize nn.Module
        self.config = config

    @abstractmethod
    def load_pretrained_weights(
        self, 
        model_path: str,
        enable_conv2d: bool = True,
        enable_vocab_split: bool = False,
        enable_vocab_split8: bool = True,
        enable_logits2: bool = True,
        enable_coreml: bool = False,
        mlp_up_split: int = 1,
        mlp_down_split: int = 1,
        enable_debug: bool = False
    ):
        """Load pretrained weights from safetensors files.
        
        Args:
            model_path: Path to directory containing model weights
            enable_conv2d: Whether to reshape weights for Conv2D operations
            enable_vocab_split: Whether to split vocabulary into 2 parts
            enable_vocab_split8: Whether to split vocabulary into 8 parts
            mlp_up_split: Number of splits for MLP up-projection
            mlp_down_split: Number of splits for MLP down-projection
            enable_debug: Whether to print debug information
        """
        pass

    @abstractmethod
    def preprocess(self):
        """Preprocessing steps before model conversion."""
        pass

    @abstractmethod
    def validate(self):
        """Validate model structure and parameters."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        pass

    def to_device(self, device):
        """Move model to specified device."""
        return self.to(device)

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        raise NotImplementedError

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        raise NotImplementedError
