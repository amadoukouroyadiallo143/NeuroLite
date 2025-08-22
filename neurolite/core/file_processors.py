"""
NeuroLite AGI - Processeurs de Fichiers Réels
Traitement réel de tous types de fichiers sans simulation
"""

import os
import io
import mimetypes
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import numpy as np

# Imports pour le traitement réel
from PIL import Image
import cv2
import librosa
import pandas as pd
import json
import xml.etree.ElementTree as ET
import docx
import PyPDF2
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import requests
from bs4 import BeautifulSoup
import zipfile
import tarfile

class FileType(Enum):
    """Types de fichiers supportés"""
    TEXT = "text"
    IMAGE = "image"  
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    DATA = "data"
    WEB = "web"
    ARCHIVE = "archive"
    CODE = "code"
    UNKNOWN = "unknown"

@dataclass
class ProcessedFile:
    """Résultat du traitement d'un fichier"""
    file_path: str
    file_type: FileType
    content: torch.Tensor
    metadata: Dict[str, Any]
    raw_data: Any
    processing_time: float
    file_size: int

class FileTypeDetector:
    """Détecteur intelligent de type de fichier"""
    
    def __init__(self):
        self.mime_to_filetype = {
            # Texte
            'text/plain': FileType.TEXT,
            'text/html': FileType.WEB,
            'text/css': FileType.CODE,
            'text/javascript': FileType.CODE,
            'application/json': FileType.DATA,
            'application/xml': FileType.DATA,
            
            # Images
            'image/jpeg': FileType.IMAGE,
            'image/png': FileType.IMAGE,
            'image/gif': FileType.IMAGE,
            'image/bmp': FileType.IMAGE,
            'image/webp': FileType.IMAGE,
            'image/tiff': FileType.IMAGE,
            
            # Audio
            'audio/wav': FileType.AUDIO,
            'audio/mp3': FileType.AUDIO,
            'audio/mpeg': FileType.AUDIO,
            'audio/ogg': FileType.AUDIO,
            'audio/flac': FileType.AUDIO,
            
            # Vidéo
            'video/mp4': FileType.VIDEO,
            'video/avi': FileType.VIDEO,
            'video/mkv': FileType.VIDEO,
            'video/mov': FileType.VIDEO,
            'video/webm': FileType.VIDEO,
            
            # Documents
            'application/pdf': FileType.DOCUMENT,
            'application/msword': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCUMENT,
            'application/vnd.ms-excel': FileType.DATA,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileType.DATA,
            
            # Archives
            'application/zip': FileType.ARCHIVE,
            'application/x-tar': FileType.ARCHIVE,
            'application/gzip': FileType.ARCHIVE,
        }
        
        self.extension_to_filetype = {
            # Code
            '.py': FileType.CODE,
            '.js': FileType.CODE,
            '.html': FileType.WEB,
            '.css': FileType.CODE,
            '.cpp': FileType.CODE,
            '.c': FileType.CODE,
            '.java': FileType.CODE,
            
            # Data
            '.csv': FileType.DATA,
            '.xlsx': FileType.DATA,
            '.json': FileType.DATA,
            '.xml': FileType.DATA,
            '.yaml': FileType.DATA,
            '.yml': FileType.DATA,
        }
    
    def detect_file_type(self, file_path: str) -> FileType:
        """Détecte le type d'un fichier"""
        if file_path.startswith('http'):
            return FileType.WEB
            
        # Vérification MIME
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type in self.mime_to_filetype:
            return self.mime_to_filetype[mime_type]
        
        # Vérification extension
        extension = Path(file_path).suffix.lower()
        if extension in self.extension_to_filetype:
            return self.extension_to_filetype[extension]
        
        # Vérification contenu (pour fichiers texte)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(100)  # Tester si c'est du texte
            return FileType.TEXT
        except:
            pass
        
        return FileType.UNKNOWN

class TextProcessor:
    """Processeur pour fichiers texte"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except:
            print(f"⚠️ Impossible de charger {model_name}, utilisation d'un processeur basique")
            self.tokenizer = None
            self.model = None
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite un fichier texte"""
        start_time = asyncio.get_event_loop().time()
        
        if file_path.startswith('http'):
            # URL
            response = requests.get(file_path)
            content_text = response.text
            file_size = len(content_text.encode())
        else:
            # Fichier local
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content_text = await f.read()
            file_size = os.path.getsize(file_path)
        
        # Traitement avec modèle ou basique
        if self.model is not None:
            inputs = self.tokenizer(content_text[:512], return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            # Embedding basique basé sur les caractéristiques du texte
            features = [
                len(content_text),
                content_text.count(' '),
                content_text.count('\n'),
                content_text.count('.'),
                content_text.count('!'),
                content_text.count('?')
            ]
            embeddings = torch.tensor([features], dtype=torch.float32)
        
        metadata = {
            'length': len(content_text),
            'lines': content_text.count('\n'),
            'words': len(content_text.split()),
            'language': 'auto-detected',  # À améliorer avec langdetect
            'encoding': 'utf-8'
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.TEXT,
            content=embeddings,
            metadata=metadata,
            raw_data=content_text,
            processing_time=processing_time,
            file_size=file_size
        )

class ImageProcessor:
    """Processeur pour images"""
    
    def __init__(self):
        try:
            from torchvision import transforms, models
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Utiliser ResNet pré-entraîné pour features
            self.feature_extractor = models.resnet50(pretrained=True)
            self.feature_extractor.fc = nn.Identity()  # Retirer la dernière couche
            self.feature_extractor.eval()
        except:
            print("⚠️ Impossible de charger torchvision, utilisation d'un processeur basique")
            self.transform = None
            self.feature_extractor = None
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite une image"""
        start_time = asyncio.get_event_loop().time()
        
        # Charger l'image
        if file_path.startswith('http'):
            response = requests.get(file_path)
            image = Image.open(io.BytesIO(response.content))
            file_size = len(response.content)
        else:
            image = Image.open(file_path)
            file_size = os.path.getsize(file_path)
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extraction de features
        if self.feature_extractor is not None:
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
        else:
            # Features basiques
            img_array = np.array(image)
            basic_features = [
                img_array.shape[0],  # height
                img_array.shape[1],  # width
                img_array.mean(),    # brightness
                img_array.std(),     # contrast
                np.mean(img_array[:,:,0]),  # red channel
                np.mean(img_array[:,:,1]),  # green channel
                np.mean(img_array[:,:,2]),  # blue channel
            ]
            features = torch.tensor([basic_features], dtype=torch.float32)
        
        metadata = {
            'width': image.size[0],
            'height': image.size[1],
            'mode': image.mode,
            'format': image.format,
            'has_transparency': 'transparency' in image.info,
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.IMAGE,
            content=features,
            metadata=metadata,
            raw_data=np.array(image),
            processing_time=processing_time,
            file_size=file_size
        )

class AudioProcessor:
    """Processeur pour audio"""
    
    def __init__(self):
        self.sample_rate = 16000
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite un fichier audio"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Charger l'audio avec librosa
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            file_size = os.path.getsize(file_path) if not file_path.startswith('http') else len(audio_data) * 4
            
            # Extraction de features audio
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Moyenner les features temporelles
            features = np.concatenate([
                mfccs.mean(axis=1),
                spectral_centroids.mean(axis=1),
                zero_crossing_rate.mean(axis=1),
                [audio_data.mean(), audio_data.std(), len(audio_data)]
            ])
            
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            print(f"⚠️ Erreur traitement audio {file_path}: {e}")
            # Features par défaut en cas d'erreur
            features_tensor = torch.zeros(1, 20)
            audio_data = np.array([])
            sr = self.sample_rate
            file_size = 0
        
        metadata = {
            'duration': len(audio_data) / sr,
            'sample_rate': sr,
            'channels': 1,
            'format': Path(file_path).suffix
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.AUDIO,
            content=features_tensor,
            metadata=metadata,
            raw_data=audio_data,
            processing_time=processing_time,
            file_size=file_size
        )

class DocumentProcessor:
    """Processeur pour documents (PDF, Word, etc.)"""
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite un document"""
        start_time = asyncio.get_event_loop().time()
        
        file_extension = Path(file_path).suffix.lower()
        file_size = os.path.getsize(file_path)
        
        try:
            if file_extension == '.pdf':
                content_text = self._process_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                content_text = self._process_word(file_path)
            else:
                content_text = "Document non supporté"
                
        except Exception as e:
            print(f"⚠️ Erreur traitement document {file_path}: {e}")
            content_text = "Erreur de traitement"
        
        # Features basées sur le texte extrait
        features = [
            len(content_text),
            content_text.count(' '),
            content_text.count('\n'),
            content_text.count('.'),
            len(content_text.split())
        ]
        features_tensor = torch.tensor([features], dtype=torch.float32)
        
        metadata = {
            'text_length': len(content_text),
            'word_count': len(content_text.split()),
            'document_type': file_extension
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.DOCUMENT,
            content=features_tensor,
            metadata=metadata,
            raw_data=content_text,
            processing_time=processing_time,
            file_size=file_size
        )
    
    def _process_pdf(self, file_path: str) -> str:
        """Extrait le texte d'un PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except:
            text = "Erreur extraction PDF"
        return text
    
    def _process_word(self, file_path: str) -> str:
        """Extrait le texte d'un document Word"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        except:
            text = "Erreur extraction Word"
        return text

class DataProcessor:
    """Processeur pour fichiers de données (CSV, JSON, etc.)"""
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite un fichier de données"""
        start_time = asyncio.get_event_loop().time()
        
        file_extension = Path(file_path).suffix.lower()
        file_size = os.path.getsize(file_path)
        
        try:
            if file_extension == '.csv':
                data, features = self._process_csv(file_path)
            elif file_extension == '.json':
                data, features = self._process_json(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                data, features = self._process_excel(file_path)
            else:
                data = "Format non supporté"
                features = torch.zeros(1, 10)
                
        except Exception as e:
            print(f"⚠️ Erreur traitement données {file_path}: {e}")
            data = "Erreur de traitement"
            features = torch.zeros(1, 10)
        
        metadata = {
            'file_format': file_extension,
            'data_type': type(data).__name__
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.DATA,
            content=features,
            metadata=metadata,
            raw_data=data,
            processing_time=processing_time,
            file_size=file_size
        )
    
    def _process_csv(self, file_path: str) -> Tuple[pd.DataFrame, torch.Tensor]:
        """Traite un fichier CSV"""
        df = pd.read_csv(file_path)
        
        # Features statistiques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_features = []
        
        if len(numeric_cols) > 0:
            stats_features.extend([
                df[numeric_cols].mean().mean(),
                df[numeric_cols].std().mean(),
                df[numeric_cols].min().min(),
                df[numeric_cols].max().max()
            ])
        else:
            stats_features.extend([0, 0, 0, 0])
        
        # Features structurelles
        structural_features = [
            len(df),  # nb rows
            len(df.columns),  # nb cols
            df.isnull().sum().sum(),  # nb nulls
            len(numeric_cols),  # nb numeric cols
            len(df.columns) - len(numeric_cols),  # nb text cols
            df.duplicated().sum()  # nb duplicates
        ]
        
        all_features = stats_features + structural_features
        features_tensor = torch.tensor([all_features], dtype=torch.float32)
        
        return df, features_tensor
    
    def _process_json(self, file_path: str) -> Tuple[Dict, torch.Tensor]:
        """Traite un fichier JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Features basées sur la structure JSON
        def count_elements(obj, depth=0):
            if depth > 10:  # Éviter récursion infinie
                return 0, 0, 0
                
            if isinstance(obj, dict):
                keys = len(obj.keys())
                values = 0
                nested = 0
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        nested += 1
                        sub_keys, sub_values, sub_nested = count_elements(v, depth+1)
                        keys += sub_keys
                        values += sub_values
                        nested += sub_nested
                    else:
                        values += 1
                return keys, values, nested
            elif isinstance(obj, list):
                items = len(obj)
                nested = 0
                for item in obj[:100]:  # Limiter pour performance
                    if isinstance(item, (dict, list)):
                        nested += 1
                return 0, items, nested
            else:
                return 0, 1, 0
        
        keys_count, values_count, nested_count = count_elements(data)
        
        features = [
            keys_count,
            values_count,
            nested_count,
            len(str(data)),  # taille sérialisée
            0, 0, 0, 0, 0, 0  # padding pour uniformité
        ]
        
        features_tensor = torch.tensor([features], dtype=torch.float32)
        
        return data, features_tensor
    
    def _process_excel(self, file_path: str) -> Tuple[pd.DataFrame, torch.Tensor]:
        """Traite un fichier Excel"""
        df = pd.read_excel(file_path)
        return self._process_csv(file_path.replace('.xlsx', '.csv'))  # Réutiliser logique CSV

class WebProcessor:
    """Processeur pour contenu web"""
    
    async def process(self, file_path: str) -> ProcessedFile:
        """Traite une URL ou fichier HTML"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if file_path.startswith('http'):
                response = requests.get(file_path, timeout=10)
                html_content = response.text
                file_size = len(html_content.encode())
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                file_size = os.path.getsize(file_path)
            
            # Parser le HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            # Features web spécifiques
            features = [
                len(text_content),
                len(soup.find_all('a')),  # nb links
                len(soup.find_all('img')),  # nb images
                len(soup.find_all('p')),  # nb paragraphs
                len(soup.find_all('div')),  # nb divs
                len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),  # nb headers
                len(soup.find_all('script')),  # nb scripts
                len(soup.find_all('style')),  # nb styles
            ]
            
            features_tensor = torch.tensor([features], dtype=torch.float32)
            
            metadata = {
                'title': soup.title.string if soup.title else "No title",
                'links_count': len(soup.find_all('a')),
                'images_count': len(soup.find_all('img')),
                'text_length': len(text_content)
            }
            
        except Exception as e:
            print(f"⚠️ Erreur traitement web {file_path}: {e}")
            features_tensor = torch.zeros(1, 8)
            text_content = "Erreur de traitement"
            metadata = {'error': str(e)}
            file_size = 0
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessedFile(
            file_path=file_path,
            file_type=FileType.WEB,
            content=features_tensor,
            metadata=metadata,
            raw_data=text_content,
            processing_time=processing_time,
            file_size=file_size
        )

class UniversalFileProcessor:
    """Processeur universel qui orchestre tous les processeurs spécialisés"""
    
    def __init__(self):
        self.detector = FileTypeDetector()
        self.processors = {
            FileType.TEXT: TextProcessor(),
            FileType.IMAGE: ImageProcessor(),
            FileType.AUDIO: AudioProcessor(),
            FileType.DOCUMENT: DocumentProcessor(),
            FileType.DATA: DataProcessor(),
            FileType.WEB: WebProcessor(),
        }
    
    async def process_file(self, file_path: str) -> ProcessedFile:
        """Traite un fichier automatiquement selon son type"""
        file_type = self.detector.detect_file_type(file_path)
        
        if file_type in self.processors:
            return await self.processors[file_type].process(file_path)
        else:
            # Processeur par défaut pour types inconnus
            return ProcessedFile(
                file_path=file_path,
                file_type=FileType.UNKNOWN,
                content=torch.zeros(1, 10),
                metadata={'error': 'Type de fichier non supporté'},
                raw_data=None,
                processing_time=0.0,
                file_size=0
            )
    
    async def process_multiple_files(self, file_paths: List[str]) -> List[ProcessedFile]:
        """Traite plusieurs fichiers en parallèle"""
        tasks = [self.process_file(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Test des processeurs
if __name__ == "__main__":
    async def test_processors():
        processor = UniversalFileProcessor()
        
        # Test avec différents types de fichiers
        test_files = [
            "test.txt",  # Sera créé pour le test
            "https://www.example.com",  # URL
            # Ajoutez d'autres fichiers de test
        ]
        
        # Créer un fichier test
        with open("test.txt", "w") as f:
            f.write("Ceci est un fichier de test pour l'AGI NeuroLite.")
        
        results = await processor.process_multiple_files(["test.txt"])
        
        for result in results:
            if isinstance(result, ProcessedFile):
                print(f"✅ {result.file_path}: {result.file_type.value}")
                print(f"   Taille: {result.file_size} bytes")
                print(f"   Temps: {result.processing_time:.3f}s")
                print(f"   Features: {result.content.shape}")
                print(f"   Metadata: {result.metadata}")
            else:
                print(f"❌ Erreur: {result}")
    
    asyncio.run(test_processors())