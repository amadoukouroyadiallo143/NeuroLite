"""
NeuroLite Structured Data Tokenizer
===================================

Tokenizer spécialisé pour données structurées.
"""

import torch
import numpy as np
import json
from typing import Any, Dict, List, Optional, Union
import logging

from ..base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizationResult, ModalityType, TokenizationStrategy

logger = logging.getLogger(__name__)

class StructuredTokenizer(BaseTokenizer):
    """Tokenizer pour données structurées."""
    
    def __init__(self, config: TokenizerConfig, modality: ModalityType = ModalityType.STRUCTURED):
        super().__init__(config, modality)
        self.strategy = TokenizationStrategy.JSON
        
    def tokenize(self, data: Any, **kwargs) -> TokenizationResult:
        """Tokenise des données structurées."""
        try:
            # Convertir en JSON string puis tokenizer
            if isinstance(data, dict):
                json_str = json.dumps(data)
            elif isinstance(data, list):
                json_str = json.dumps(data)
            else:
                json_str = str(data)
            
            # Tokenization simple par caractères
            tokens = [ord(c) for c in json_str]
            
            return TokenizationResult(
                tokens=tokens,
                modality=ModalityType.STRUCTURED,
                strategy=self.strategy,
                vocab_size=self.get_vocab_size(),
                sequence_length=len(tokens)
            )
        except Exception as e:
            return TokenizationResult(tokens=[], metadata={'error': str(e)})
    
    def detokenize(self, tokens: Union[torch.Tensor, np.ndarray, List[int]], **kwargs) -> str:
        """Reconstruit les données structurées."""
        try:
            chars = [chr(t) for t in tokens if 32 <= t <= 126]
            return ''.join(chars)
        except:
            return "{}"
    
    def get_vocab_size(self) -> int:
        return 256  # ASCII characters
    
    def get_supported_strategies(self) -> List[TokenizationStrategy]:
        return [TokenizationStrategy.JSON, TokenizationStrategy.CSV, TokenizationStrategy.XML]