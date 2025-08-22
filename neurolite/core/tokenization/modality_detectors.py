"""
NeuroLite Modality Detection System
==================================

Système de détection automatique des modalités de données.
Utilise des heuristiques avancées pour identifier le type de données.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

# Imports optionnels pour détection avancée
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base_tokenizer import ModalityType

logger = logging.getLogger(__name__)

class ModalityDetector:
    """Détecteur intelligent de modalités avec heuristiques avancées."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialise le détecteur de modalités.
        
        Args:
            confidence_threshold: Seuil de confiance pour la détection
        """
        self.confidence_threshold = confidence_threshold
        self.detection_history = []
        
    def detect(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[ModalityType, float]:
        """
        Détecte la modalité des données avec score de confiance.
        
        Args:
            data: Données à analyser
            context: Contexte additionnel (nom fichier, métadonnées, etc.)
            
        Returns:
            Tuple[ModalityType, float]: (modalité détectée, score de confiance)
        """
        start_time = time.time()
        
        try:
            # Détection par type Python
            modality, confidence = self._detect_by_type(data)
            
            # Si confiance faible, essayer détection par contenu
            if confidence < self.confidence_threshold:
                content_modality, content_confidence = self._detect_by_content(data)
                if content_confidence > confidence:
                    modality, confidence = content_modality, content_confidence
            
            # Si contexte disponible, affiner la détection
            if context and confidence < self.confidence_threshold:
                context_modality, context_confidence = self._detect_by_context(data, context)
                if context_confidence > confidence:
                    modality, confidence = context_modality, context_confidence
            
            # Enregistrer dans l'historique
            detection_time = (time.time() - start_time) * 1000
            self.detection_history.append({
                'modality': modality,
                'confidence': confidence,
                'detection_time_ms': detection_time,
                'data_type': type(data).__name__,
                'data_shape': getattr(data, 'shape', None),
                'context': context
            })
            
            logger.debug(f"Modalité détectée: {modality.value} (confiance: {confidence:.3f})")
            return modality, confidence
            
        except Exception as e:
            logger.error(f"Erreur détection modalité: {e}")
            return ModalityType.UNKNOWN, 0.0
    
    def _detect_by_type(self, data: Any) -> Tuple[ModalityType, float]:
        """Détection basée sur le type Python."""
        
        # Texte
        if isinstance(data, str):
            return ModalityType.TEXT, 0.9
        
        # Tenseurs et arrays
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return self._detect_tensor_modality(data)
        
        # Images PIL
        if hasattr(data, 'mode') and hasattr(data, 'size'):  # PIL Image
            return ModalityType.IMAGE, 0.95
        
        # Structures de données
        if isinstance(data, (dict, list)):
            return self._detect_structured_modality(data)
        
        # Fichiers
        if isinstance(data, (Path, str)) and Path(data).exists():
            return self._detect_file_modality(data)
        
        return ModalityType.UNKNOWN, 0.0
    
    def _detect_tensor_modality(self, data: Union[torch.Tensor, np.ndarray]) -> Tuple[ModalityType, float]:
        """Détection spécialisée pour tenseurs/arrays."""
        shape = data.shape
        ndim = len(shape)
        
        # Texte: 1D ou 2D avec dimension de vocabulaire
        if ndim == 1 or (ndim == 2 and shape[1] < 100000):
            return ModalityType.TEXT, 0.7
        
        # Image: 2D (grayscale) ou 3D (RGB/RGBA)
        if ndim == 2 or (ndim == 3 and shape[-1] in [1, 3, 4]):
            # Vérifier si les valeurs sont dans la plage d'une image
            if data.dtype in [np.uint8, torch.uint8] or (0 <= data.min() and data.max() <= 255):
                return ModalityType.IMAGE, 0.8
        
        # Image batch: 4D (B, H, W, C) ou (B, C, H, W)
        if ndim == 4:
            if shape[-1] in [1, 3, 4] or shape[1] in [1, 3, 4]:
                return ModalityType.IMAGE, 0.8
        
        # Audio: 1D (waveform) ou 2D (spectrogram)
        if ndim == 1 and len(shape[0]) > 1000:  # Long signal 1D
            return ModalityType.AUDIO, 0.7
        if ndim == 2 and shape[0] < shape[1]:  # Forme typique spectrogram
            return ModalityType.AUDIO, 0.7
        
        # Vidéo: 4D ou 5D
        if ndim >= 4:
            # (T, H, W, C) ou (B, T, H, W, C)
            return ModalityType.VIDEO, 0.6
        
        return ModalityType.UNKNOWN, 0.0
    
    def _detect_structured_modality(self, data: Union[dict, list]) -> Tuple[ModalityType, float]:
        """Détection pour données structurées."""
        
        if isinstance(data, dict):
            # JSON-like structure
            try:
                json.dumps(data)  # Test si sérialisable en JSON
                return ModalityType.STRUCTURED, 0.8
            except (TypeError, ValueError):
                pass
        
        if isinstance(data, list):
            if len(data) > 0:
                first_item = data[0]
                # Liste de dictionnaires (CSV-like)
                if isinstance(first_item, dict):
                    return ModalityType.STRUCTURED, 0.8
                # Liste de listes (matrice)
                if isinstance(first_item, list):
                    return ModalityType.STRUCTURED, 0.7
        
        return ModalityType.STRUCTURED, 0.5
    
    def _detect_file_modality(self, file_path: Union[str, Path]) -> Tuple[ModalityType, float]:
        """Détection basée sur l'extension de fichier."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Extensions texte
        text_exts = {'.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.html', '.csv'}
        if extension in text_exts:
            return ModalityType.TEXT, 0.9
        
        # Extensions image
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        if extension in image_exts:
            return ModalityType.IMAGE, 0.9
        
        # Extensions audio
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        if extension in audio_exts:
            return ModalityType.AUDIO, 0.9
        
        # Extensions vidéo
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        if extension in video_exts:
            return ModalityType.VIDEO, 0.9
        
        # Extensions données structurées
        struct_exts = {'.csv', '.json', '.xml', '.yaml', '.yml', '.parquet'}
        if extension in struct_exts:
            return ModalityType.STRUCTURED, 0.9
        
        return ModalityType.UNKNOWN, 0.0
    
    def _detect_by_content(self, data: Any) -> Tuple[ModalityType, float]:
        """Détection basée sur l'analyse du contenu."""
        
        # Pour les chaînes de caractères
        if isinstance(data, str):
            return self._analyze_text_content(data)
        
        # Pour les données binaires
        if isinstance(data, bytes):
            return self._analyze_binary_content(data)
        
        return ModalityType.UNKNOWN, 0.0
    
    def _analyze_text_content(self, text: str) -> Tuple[ModalityType, float]:
        """Analyse le contenu textuel pour affiner la détection."""
        
        # JSON
        try:
            json.loads(text)
            return ModalityType.STRUCTURED, 0.8
        except (json.JSONDecodeError, ValueError):
            pass
        
        # XML
        try:
            ET.fromstring(text)
            return ModalityType.STRUCTURED, 0.8
        except ET.ParseError:
            pass
        
        # CSV (heuristique simple)
        lines = text.split('\n')[:10]  # Premiers lignes
        if len(lines) > 1:
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                if all(delimiter in line for line in lines if line.strip()):
                    return ModalityType.STRUCTURED, 0.7
        
        # Texte libre par défaut
        return ModalityType.TEXT, 0.6
    
    def _analyze_binary_content(self, data: bytes) -> Tuple[ModalityType, float]:
        """Analyse le contenu binaire (signatures de fichiers)."""
        
        # Signatures d'images
        image_signatures = {
            b'\xFF\xD8\xFF': ModalityType.IMAGE,  # JPEG
            b'\x89PNG\r\n\x1a\n': ModalityType.IMAGE,  # PNG
            b'GIF8': ModalityType.IMAGE,  # GIF
            b'BM': ModalityType.IMAGE,  # BMP
        }
        
        # Signatures audio
        audio_signatures = {
            b'RIFF': ModalityType.AUDIO,  # WAV
            b'ID3': ModalityType.AUDIO,  # MP3
            b'\xFF\xFB': ModalityType.AUDIO,  # MP3
            b'fLaC': ModalityType.AUDIO,  # FLAC
        }
        
        # Signatures vidéo
        video_signatures = {
            b'\x00\x00\x00\x20ftypmp4': ModalityType.VIDEO,  # MP4
            b'RIFF....AVI ': ModalityType.VIDEO,  # AVI
        }
        
        # Vérifier les signatures
        for signature, modality in {**image_signatures, **audio_signatures, **video_signatures}.items():
            if data.startswith(signature):
                return modality, 0.9
        
        return ModalityType.UNKNOWN, 0.0
    
    def _detect_by_context(self, data: Any, context: Dict[str, Any]) -> Tuple[ModalityType, float]:
        """Détection basée sur le contexte fourni."""
        
        # Nom de fichier dans le contexte
        if 'filename' in context:
            return self._detect_file_modality(context['filename'])
        
        # Type MIME
        if 'mime_type' in context:
            mime = context['mime_type'].lower()
            if mime.startswith('text/'):
                return ModalityType.TEXT, 0.8
            elif mime.startswith('image/'):
                return ModalityType.IMAGE, 0.8
            elif mime.startswith('audio/'):
                return ModalityType.AUDIO, 0.8
            elif mime.startswith('video/'):
                return ModalityType.VIDEO, 0.8
        
        # Modalité explicite
        if 'modality' in context:
            try:
                modality = ModalityType(context['modality'])
                return modality, 0.9
            except ValueError:
                pass
        
        return ModalityType.UNKNOWN, 0.0
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de détection."""
        if not self.detection_history:
            return {}
        
        total = len(self.detection_history)
        by_modality = {}
        avg_confidence = 0.0
        avg_time = 0.0
        
        for detection in self.detection_history:
            modality = detection['modality'].value
            by_modality[modality] = by_modality.get(modality, 0) + 1
            avg_confidence += detection['confidence']
            avg_time += detection['detection_time_ms']
        
        return {
            'total_detections': total,
            'average_confidence': avg_confidence / total,
            'average_time_ms': avg_time / total,
            'by_modality': by_modality,
            'success_rate': sum(1 for d in self.detection_history if d['confidence'] >= self.confidence_threshold) / total
        }

# Instance globale pour usage facile
_global_detector = ModalityDetector()

def detect_modality(data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[ModalityType, float]:
    """
    Fonction utilitaire pour détecter la modalité des données.
    
    Args:
        data: Données à analyser
        context: Contexte optionnel
        
    Returns:
        Tuple[ModalityType, float]: (modalité, confiance)
    """
    return _global_detector.detect(data, context)

def get_detector_statistics() -> Dict[str, Any]:
    """Retourne les statistiques du détecteur global."""
    return _global_detector.get_detection_statistics()