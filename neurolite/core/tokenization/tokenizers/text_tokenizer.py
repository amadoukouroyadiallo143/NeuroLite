"""
NeuroLite Text Tokenizer
=======================

Tokenizer spécialisé pour le texte avec support de multiples stratégies:
- BPE (Byte Pair Encoding)
- WordPiece 
- SentencePiece
- Character Level

Intégration avec HuggingFace Transformers pour compatibilité maximale.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
import time
import re
from pathlib import Path

from ..base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizationResult, ModalityType, TokenizationStrategy

# Imports conditionnels pour HuggingFace
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = None

logger = logging.getLogger(__name__)

class TextTokenizer(BaseTokenizer):
    """Tokenizer avancé pour texte avec multiples stratégies."""
    
    def __init__(self, config: TokenizerConfig, modality: ModalityType = ModalityType.TEXT):
        """
        Initialise le tokenizer de texte.
        
        Args:
            config: Configuration du tokenizer
            modality: Type de modalité (TEXT par défaut)
        """
        super().__init__(config, modality)
        
        # Configuration spécifique au texte
        text_config = self.modality_config
        self.strategy = TokenizationStrategy(text_config.get('strategy', 'bpe'))
        self.model_name = text_config.get('model_name', 'bert-base-uncased')
        self.custom_vocab_file = text_config.get('custom_vocab_file', None)
        self.special_tokens = text_config.get('special_tokens', ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        # Paramètres de tokenization
        self.max_length = min(config.max_sequence_length, text_config.get('max_length', 512))
        self.add_special_tokens = text_config.get('add_special_tokens', True)
        self.return_attention_mask = text_config.get('return_attention_mask', True)
        
        # Initialiser le tokenizer selon la stratégie
        self.tokenizer = self._initialize_tokenizer()
        
        # Statistiques spécifiques
        self.total_characters = 0
        self.total_words = 0
        self.total_sentences = 0
        
        logger.info(f"TextTokenizer initialisé avec stratégie {self.strategy.value}")
    
    def _initialize_tokenizer(self) -> Optional[Any]:
        """Initialise le tokenizer selon la stratégie choisie."""
        
        try:
            if self.strategy == TokenizationStrategy.BPE:
                return self._init_bpe_tokenizer()
            elif self.strategy == TokenizationStrategy.WORDPIECE:
                return self._init_wordpiece_tokenizer()
            elif self.strategy == TokenizationStrategy.SENTENCEPIECE:
                return self._init_sentencepiece_tokenizer()
            elif self.strategy == TokenizationStrategy.CHAR_LEVEL:
                return self._init_char_tokenizer()
            else:
                logger.warning(f"Stratégie {self.strategy.value} non supportée, utilisation BPE par défaut")
                return self._init_bpe_tokenizer()
                
        except Exception as e:
            logger.error(f"Erreur initialisation tokenizer: {e}")
            return self._init_fallback_tokenizer()
    
    def _init_bpe_tokenizer(self) -> Any:
        """Initialise un tokenizer BPE (GPT-style)."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Essayer d'utiliser un modèle GPT pour BPE
                model_name = self.modality_config.get('bpe_model', 'gpt2')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Configurer les tokens spéciaux
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                return tokenizer
            except Exception as e:
                logger.warning(f"Impossible de charger le tokenizer BPE HuggingFace: {e}")
        
        # Fallback vers implémentation simple
        return self._create_simple_bpe_tokenizer()
    
    def _init_wordpiece_tokenizer(self) -> Any:
        """Initialise un tokenizer WordPiece (BERT-style)."""
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.modality_config.get('wordpiece_model', 'bert-base-uncased')
                return AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Impossible de charger le tokenizer WordPiece: {e}")
        
        return self._create_simple_wordpiece_tokenizer()
    
    def _init_sentencepiece_tokenizer(self) -> Any:
        """Initialise un tokenizer SentencePiece (T5-style)."""
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.modality_config.get('sentencepiece_model', 't5-base')
                return AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Impossible de charger le tokenizer SentencePiece: {e}")
        
        return self._create_simple_sentencepiece_tokenizer()
    
    def _init_char_tokenizer(self) -> Any:
        """Initialise un tokenizer au niveau caractère."""
        return self._create_char_level_tokenizer()
    
    def _init_fallback_tokenizer(self) -> Any:
        """Tokenizer de fallback simple."""
        return self._create_simple_word_tokenizer()
    
    def tokenize(self, data: str, **kwargs) -> TokenizationResult:
        """
        Tokenise un texte.
        
        Args:
            data: Texte à tokeniser
            **kwargs: Arguments additionnels
            
        Returns:
            TokenizationResult: Résultat de la tokenization
        """
        start_time = time.time()
        
        try:
            # Validation des données
            if not isinstance(data, str):
                data = str(data)
            
            # Préprocessing optionnel
            if kwargs.get('preprocess', True):
                data = self._preprocess_text(data)
            
            # Statistiques
            self.total_characters += len(data)
            self.total_words += len(data.split())
            self.total_sentences += len(re.split(r'[.!?]+', data))
            
            # Tokenization selon le type de tokenizer
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, PreTrainedTokenizer):
                result = self._tokenize_with_transformers(data, **kwargs)
            else:
                result = self._tokenize_with_custom(data, **kwargs)
            
            # Post-traitement
            result = self._postprocess_result(result, data)
            
            # Métriques
            processing_time = (time.time() - start_time) * 1000
            result.tokenization_time_ms = processing_time
            
            # Calculer le ratio de compression
            if result.sequence_length > 0:
                result.compression_ratio = len(data) / result.sequence_length
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur tokenization texte: {e}")
            return TokenizationResult(
                tokens=[],
                modality=ModalityType.TEXT,
                tokenization_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _tokenize_with_transformers(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenise avec un tokenizer HuggingFace."""
        
        # Paramètres de tokenization
        tokenizer_kwargs = {
            'text': text,
            'max_length': kwargs.get('max_length', self.max_length),
            'padding': kwargs.get('padding', 'max_length' if self.config.padding_strategy == 'max_length' else False),
            'truncation': kwargs.get('truncation', self.config.truncation),
            'return_tensors': kwargs.get('return_tensors', self.config.return_tensors),
            'add_special_tokens': kwargs.get('add_special_tokens', self.add_special_tokens),
            'return_attention_mask': kwargs.get('return_attention_mask', self.return_attention_mask),
            'return_token_type_ids': kwargs.get('return_token_type_ids', False)
        }
        
        # Tokeniser
        encoded = self.tokenizer(**tokenizer_kwargs)
        
        # Extraire les composants
        tokens = encoded['input_ids']
        attention_mask = encoded.get('attention_mask', None)
        
        # Convertir en format approprié
        if self.config.return_tensors == 'torch':
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
        elif self.config.return_tensors == 'numpy':
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.numpy()
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.numpy()
        elif self.config.return_tensors == 'list':
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            if attention_mask is not None and hasattr(attention_mask, 'tolist'):
                attention_mask = attention_mask.tolist()
        
        return TokenizationResult(
            tokens=tokens,
            attention_mask=attention_mask,
            modality=ModalityType.TEXT,
            strategy=self.strategy,
            vocab_size=self.tokenizer.vocab_size,
            sequence_length=len(tokens) if hasattr(tokens, '__len__') else tokens.shape[-1],
            metadata={
                'tokenizer_type': 'transformers',
                'model_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
                'special_tokens_added': self.add_special_tokens
            }
        )
    
    def _tokenize_with_custom(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenise avec un tokenizer personnalisé."""
        
        if hasattr(self.tokenizer, 'encode'):
            # Tokenizer personnalisé avec méthode encode
            tokens = self.tokenizer.encode(text)
        else:
            # Fallback simple
            tokens = self._simple_tokenize(text)
        
        # Appliquer padding/truncation si nécessaire
        max_length = kwargs.get('max_length', self.max_length)
        
        if len(tokens) > max_length and self.config.truncation:
            tokens = tokens[:max_length]
        
        if len(tokens) < max_length and self.config.padding_strategy == 'max_length':
            pad_token = kwargs.get('pad_token_id', 0)
            tokens.extend([pad_token] * (max_length - len(tokens)))
        
        # Créer attention mask
        attention_mask = None
        if self.return_attention_mask:
            attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        return TokenizationResult(
            tokens=tokens,
            attention_mask=attention_mask,
            modality=ModalityType.TEXT,
            strategy=self.strategy,
            vocab_size=getattr(self.tokenizer, 'vocab_size', len(tokens)),
            sequence_length=len(tokens),
            metadata={
                'tokenizer_type': 'custom',
                'strategy': self.strategy.value
            }
        )
    
    def detokenize(self, tokens: Union[torch.Tensor, np.ndarray, List[int]], **kwargs) -> str:
        """
        Reconstruit le texte à partir des tokens.
        
        Args:
            tokens: Tokens à détokeniser
            **kwargs: Arguments additionnels
            
        Returns:
            str: Texte reconstruit
        """
        try:
            # Convertir en format approprié
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            elif isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            
            # Détokeniser selon le type
            if TRANSFORMERS_AVAILABLE and isinstance(self.tokenizer, PreTrainedTokenizer):
                text = self.tokenizer.decode(tokens, skip_special_tokens=kwargs.get('skip_special_tokens', True))
            else:
                text = self._custom_detokenize(tokens, **kwargs)
            
            return text
            
        except Exception as e:
            logger.error(f"Erreur détokenization: {e}")
            return ""
    
    def get_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, '__len__'):
            return len(self.tokenizer)
        else:
            return self.config.vocab_size
    
    def get_supported_strategies(self) -> List[TokenizationStrategy]:
        """Retourne les stratégies supportées."""
        return [
            TokenizationStrategy.BPE,
            TokenizationStrategy.WORDPIECE,
            TokenizationStrategy.SENTENCEPIECE,
            TokenizationStrategy.CHAR_LEVEL
        ]
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing optionnel du texte."""
        # Nettoyage de base
        text = text.strip()
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Autres preprocessing selon la configuration
        if self.modality_config.get('lowercase', False):
            text = text.lower()
        
        if self.modality_config.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _postprocess_result(self, result: TokenizationResult, original_text: str) -> TokenizationResult:
        """Post-traitement du résultat."""
        
        # Ajouter métadonnées sur le texte original
        result.metadata.update({
            'original_length': len(original_text),
            'original_words': len(original_text.split()),
            'compression_achieved': len(original_text) / max(1, result.sequence_length)
        })
        
        return result
    
    # ============ Implémentations simples de fallback ============
    
    def _create_simple_bpe_tokenizer(self):
        """Crée un tokenizer BPE simple."""
        class SimpleBPETokenizer:
            def __init__(self, vocab_size=10000):
                self.vocab_size = vocab_size
                self.word_to_id = {'<pad>': 0, '<unk>': 1}
                self.id_to_word = {0: '<pad>', 1: '<unk>'}
                self.current_id = 2
            
            def encode(self, text):
                words = text.split()
                tokens = []
                for word in words:
                    if word not in self.word_to_id:
                        if self.current_id < self.vocab_size:
                            self.word_to_id[word] = self.current_id
                            self.id_to_word[self.current_id] = word
                            self.current_id += 1
                            tokens.append(self.word_to_id[word])
                        else:
                            tokens.append(1)  # <unk>
                    else:
                        tokens.append(self.word_to_id[word])
                return tokens
            
            def decode(self, tokens):
                words = [self.id_to_word.get(token, '<unk>') for token in tokens]
                return ' '.join(word for word in words if word not in ['<pad>', '<unk>'])
        
        return SimpleBPETokenizer(self.config.vocab_size)
    
    def _create_simple_wordpiece_tokenizer(self):
        """Crée un tokenizer WordPiece simple."""
        return self._create_simple_bpe_tokenizer()  # Fallback vers BPE
    
    def _create_simple_sentencepiece_tokenizer(self):
        """Crée un tokenizer SentencePiece simple."""
        return self._create_simple_bpe_tokenizer()  # Fallback vers BPE
    
    def _create_char_level_tokenizer(self):
        """Crée un tokenizer au niveau caractère."""
        class CharTokenizer:
            def __init__(self):
                # Créer vocabulaire des caractères ASCII + spéciaux
                self.vocab = ['<pad>', '<unk>'] + [chr(i) for i in range(32, 127)]
                self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
                self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
                self.vocab_size = len(self.vocab)
            
            def encode(self, text):
                return [self.char_to_id.get(char, 1) for char in text]  # 1 = <unk>
            
            def decode(self, tokens):
                return ''.join(self.id_to_char.get(token, '') for token in tokens)
        
        return CharTokenizer()
    
    def _create_simple_word_tokenizer(self):
        """Crée un tokenizer simple par mots."""
        return self._create_simple_bpe_tokenizer()
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Tokenization très simple en fallback."""
        words = text.split()
        # Hasher simple pour assigner des IDs
        return [hash(word) % self.config.vocab_size for word in words]
    
    def _custom_detokenize(self, tokens: List[int], **kwargs) -> str:
        """Détokenization personnalisée."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        else:
            # Fallback très simple
            return ' '.join(f"token_{token}" for token in tokens)