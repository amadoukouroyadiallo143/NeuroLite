"""
Script de benchmark avancé pour la comparaison d'architectures de modèles de langage.

Ce script permet de comparer les performances de différentes architectures en termes de :
- Empreinte mémoire
- Temps d'inférence
- Complexité computationnelle
- Mise à l'échelle avec la longueur de séquence
- Comparaison avec des modèles pré-entraînés
- Analyse des FLOPS et du débit

Utilisation :
    python benchmark_comparison.py --batch_size 8 --seq_length 128 --num_runs 100
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tabulate import tabulate
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Importer NeuroLite
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurolite import NeuroLiteConfig, NeuroLiteModel

# Configuration des styles pour les graphiques
plt.style.use('seaborn-v0_8')  # Style moderne compatible avec seaborn
sns.set_theme(style="whitegrid")  # Utilisation du thème seaborn moderne
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.3

# Constantes
RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration commune pour tous les modèles"""
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = None
    dropout: float = 0.1
    vocab_size: int = 30000
    max_seq_length: int = 512
    
    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = self.hidden_size * 4

@dataclass
class BenchmarkResult:
    """Classe pour stocker les résultats d'un benchmark"""
    model_name: str
    parameters: int
    inference_time: float
    memory_usage: float
    sequence_length: int
    batch_size: int
    flops: float = 0.0
    throughput: float = 0.0
    output_shape: Any = None
    config: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire"""
        result = asdict(self)
        # Convertir les types non sérialisables en chaînes
        if isinstance(result.get('output_shape'), (torch.Size, tuple)):
            result['output_shape'] = str(result['output_shape'])
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Crée un BenchmarkResult à partir d'un dictionnaire"""
        return cls(**data)

class BenchmarkResults:
    """Classe pour gérer une collection de résultats de benchmark"""
    
    def __init__(self, results: Optional[List[BenchmarkResult]] = None):
        self.results = results or []
    
    def add(self, result: BenchmarkResult):
        """Ajoute un résultat à la collection"""
        self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les résultats en DataFrame pandas"""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def save(self, filename: str):
        """Sauvegarde les résultats dans un fichier JSON"""
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'BenchmarkResults':
        """Charge les résultats à partir d'un fichier JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls([BenchmarkResult.from_dict(d) for d in data])

def measure_memory_usage(func):
    """Mesure l'utilisation de mémoire avant et après l'exécution d'une fonction"""
    def wrapper(*args, **kwargs):
        # Mesurer la mémoire avant
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # en Mo
        
        # Exécuter la fonction
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = time.time() - start_time
        
        # Mesurer la mémoire après
        mem_after = process.memory_info().rss / 1024 / 1024  # en Mo
        mem_used = mem_after - mem_before
        
        # Ajouter les mesures au résultat
        if isinstance(result, dict):
            result['memory_mb'] = mem_used
            result['execution_time'] = exec_time
        
        return result, mem_used
    return wrapper

class SimpleTransformer(nn.Module):
    """Implémentation simplifiée d'un Transformer pour la comparaison"""
    
    def __init__(self, hidden_size=256, num_layers=4, num_heads=4):
        super().__init__()
        
        # Configuration simplifiée
        self.embedding = nn.Linear(768, hidden_size)  # Simuler un embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer(x)
        x = self.output(x)
        return x


@measure_memory_usage
def run_neurolite(input_size: int, seq_length: int, batch_size: int, device: torch.device) -> Tuple[Dict[str, Any], float]:
    """
    Exécute et mesure les performances de NeuroLite.
    
    Args:
        input_size: Taille de la couche d'entrée
        seq_length: Longueur de la séquence d'entrée
        batch_size: Taille du lot
        device: Appareil à utiliser (CPU/GPU)
        
    Returns:
        Tuple contenant (résultats, utilisation_mémoire)
    """
    # Dictionnaire de résultat par défaut
    default_result = {
        "model_name": "NeuroLite",
        "param_count": 0,
        "inference_time": 0.0,
        "output_shape": "error",
        "sequence_length": seq_length,
        "batch_size": batch_size,
        "flops": 0.0,
        "throughput": 0.0,
        "memory_usage": 0.0,
        "parameters": 0
    }
    
    try:
        # Utiliser une configuration plus petite pour le benchmark
        config = NeuroLiteConfig.tiny()
        config.hidden_size = input_size
        
        # Configurer le type de projection d'entrée pour accepter des identifiants de jetons
        config.input_projection_type = "ngram_hash"  # Utiliser TokenizedMinHashProjection
        config.vocab_size = 10000  # Doit correspondre à la plage de valeurs d'entrée
        
        # Créer le modèle
        model = NeuroLiteModel(config).to(device)
        model.eval()  # Passer en mode évaluation
        
        # Créer des données d'entrée simulées au format attendu par NeuroLite
        input_ids = torch.randint(
            low=0,
            high=10000,  # Taille arbitraire du vocabulaire
            size=(batch_size, seq_length),
            device=device
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        
        # Préparer les entrées multimodales
        multimodal_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Mesurer le temps d'inférence
        start_time = time.time()
        with torch.no_grad():
            outputs = model(multimodal_inputs=multimodal_inputs)
        inference_time = time.time() - start_time
        
        # Gérer différents formats de sortie
        output_shape = "unknown"
        try:
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    output_shape = str(tuple(outputs['logits'].shape))
                elif 'last_hidden_state' in outputs:
                    output_shape = str(tuple(outputs['last_hidden_state'].shape))
                else:
                    # Prendre le premier tenseur du dictionnaire
                    output_tensor = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
                    if output_tensor is not None:
                        output_shape = str(tuple(output_tensor.shape))
            elif hasattr(outputs, 'shape'):
                output_shape = str(tuple(outputs.shape))
        except Exception as e:
            print(f"Erreur lors de la détermination de la forme de sortie: {str(e)}")
            output_shape = f"error: {str(e)[:100]}"
        
        # Calculer l'utilisation mémoire après l'exécution
        memory_usage = measure_memory_usage(lambda: None)  # Mesurer la mémoire actuelle
        
        # Créer le dictionnaire de résultats
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        result = {
            **default_result,  # Utiliser les valeurs par défaut comme base
            "param_count": param_count,
            "inference_time": inference_time,
            "output_shape": output_shape,
            "memory_usage": memory_usage,
            "parameters": param_count
        }
        
        return result, memory_usage
        
    except Exception as e:
        error_msg = f"Erreur dans run_neurolite: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Mettre à jour le message d'erreur
        default_result["output_shape"] = f"error: {str(e)[:100]}"
        return default_result, 0.0

@measure_memory_usage
def run_transformer(input_size: int, seq_length: int, batch_size: int, device: torch.device) -> Tuple[Dict[str, Any], float]:
    """
    Exécute et mesure les performances d'un Transformer standard.
    
    Args:
        input_size: Taille de la couche d'entrée
        seq_length: Longueur de la séquence d'entrée
        batch_size: Taille du lot
        device: Appareil à utiliser (CPU/GPU)
        
    Returns:
        Tuple contenant (résultats, utilisation_mémoire)
    """
    # Dictionnaire de résultat par défaut
    default_result = {
        "model_name": "Transformer",
        "param_count": 0,
        "inference_time": 0.0,
        "output_shape": "error",
        "sequence_length": seq_length,
        "batch_size": batch_size,
        "flops": 0.0,
        "throughput": 0.0,
        "memory_usage": 0.0,
        "parameters": 0
    }
    
    try:
        # Créer un modèle Transformer simple
        model = SimpleTransformer(
            hidden_size=input_size,
            num_layers=4,
            num_heads=4
        ).to(device)
        model.eval()
        
        # Créer des données d'entrée simulées
        dummy_input = torch.randn(batch_size, seq_length, 768).to(device)  # 768 est la taille d'entrée standard
        
        # Mesurer le temps d'inférence
        start_time = time.time()
        with torch.no_grad():
            outputs = model(dummy_input)
        inference_time = time.time() - start_time
        
        # Déterminer la forme de sortie
        output_shape = "unknown"
        try:
            output_shape = str(tuple(outputs.shape)) if hasattr(outputs, 'shape') else str(type(outputs))
        except Exception as e:
            print(f"Erreur lors de la détermination de la forme de sortie: {str(e)}")
            output_shape = f"error: {str(e)[:100]}"
        
        # Calculer l'utilisation mémoire après l'exécution
        memory_usage = measure_memory_usage(lambda: None)
        
        # Créer le dictionnaire de résultats
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        result = {
            **default_result,  # Utiliser les valeurs par défaut comme base
            "param_count": param_count,
            "inference_time": inference_time,
            "output_shape": output_shape,
            "memory_usage": memory_usage,
            "parameters": param_count
        }
        
        return result, memory_usage
        
    except Exception as e:
        error_msg = f"Erreur dans run_transformer: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Mettre à jour le message d'erreur
        default_result["output_shape"] = f"error: {str(e)[:100]}"
        return default_result, 0.0

@measure_memory_usage
def run_huggingface(model_name: str, seq_length: int, batch_size: int, device: torch.device) -> Tuple[Dict[str, Any], float]:
    """
    Exécute et mesure les performances d'un modèle HuggingFace.
    
    Args:
        model_name: Nom du modèle HuggingFace
        seq_length: Longueur de la séquence d'entrée
        batch_size: Taille du lot
        device: Appareil à utiliser (CPU/GPU)
        
    Returns:
        Tuple contenant (résultats, utilisation_mémoire)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        
        # Préparer l'entrée avec le format attendu
        input_ids = torch.randint(
            low=0, 
            high=len(tokenizer) - 1,  # Utiliser la taille du vocabulaire du tokenizer
            size=(batch_size, seq_length),
            device=device
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        
        # Mesurer le temps d'inférence
        start_time = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        inference_time = time.time() - start_time
        
        # Calculer l'utilisation mémoire après l'exécution
        memory_usage = measure_memory_usage(lambda: None)
        
        # Gérer différents formats de sortie
        if hasattr(outputs, 'last_hidden_state'):
            output_shape = tuple(outputs.last_hidden_state.shape)
        elif hasattr(outputs, 'logits'):
            output_shape = tuple(outputs.logits.shape)
        else:
            output_shape = str(type(outputs))
        
        # Créer le dictionnaire de résultats
        result = {
            "model_name": model_name,
            "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "inference_time": inference_time,
            "output_shape": output_shape,
            "sequence_length": seq_length,
            "batch_size": batch_size,
            "flops": 0.0,  # Serait calculé avec measure_flops si nécessaire
            "throughput": 0.0,  # Serait calculé avec measure_throughput si nécessaire
            "memory_usage": memory_usage,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        return result, memory_usage
        
    except Exception as e:
        error_msg = f"Erreur dans run_huggingface avec {model_name}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Retourner un résultat d'erreur avec les métriques minimales
        error_result = {
            "model_name": f"{model_name} - ERREUR",
            "param_count": 0,
            "inference_time": 0.0,
            "output_shape": f"Erreur: {str(e)}",
            "sequence_length": seq_length,
            "batch_size": batch_size,
            "flops": 0.0,
            "throughput": 0.0,
            "memory_usage": 0.0,
            "parameters": 0
        }
        
        return error_result, 0.0

def sequence_length_scaling_test(models, seq_lengths, batch_size, device):
    """Teste l'évolution du temps d'inférence avec la longueur de séquence"""
    results = {}
    
    for model_name, model_fn in models.items():
        results[model_name] = {
            'seq_lengths': [],
            'times': [],
            'memory': []
        }
        
        for seq_len in seq_lengths:
            print(f"Test de {model_name} avec seq_length={seq_len}")
            try:
                result, memory = model_fn(seq_len, seq_len, batch_size, device)
                results[model_name]['seq_lengths'].append(seq_len)
                results[model_name]['times'].append(result['inference_time'])
                results[model_name]['memory'].append(memory)
            except Exception as e:
                print(f"Erreur avec {model_name} et seq_len={seq_len}: {str(e)}")
    
    return results

def measure_flops(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> float:
    """
    Mesure les FLOPS d'un modèle pour une entrée donnée en utilisant torch.profiler
    
    Args:
        model: Le modèle à analyser
        input_shape: La forme de l'entrée (batch_size, seq_len, ...)
        device: L'appareil à utiliser ('cuda' ou 'cpu')
        
    Returns:
        Nombre de FLOPS (opérations en virgule flottante)
    """
    try:
        from thop import profile, clever_format
        
        # Créer une entrée factice
        dummy_input = torch.randn(input_shape).to(device)
        
        # Compter les opérations
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # 1 MAC = 2 FLOPS
        
        return flops
        
    except ImportError:
        print("thop n'est pas installé. Installation avec: pip install thop")
        return 0.0

def measure_throughput(model_or_func, input_shape: tuple, num_runs: int = 100, warmup: int = 10, 
                      device: torch.device = None, is_func: bool = False, **kwargs) -> float:
    """
    Mesure le débit d'inférence (exemples par seconde)
    
    Args:
        model_or_func: Le modèle ou la fonction à évaluer
        input_shape: La forme de l'entrée (batch_size, seq_len, ...)
        num_runs: Nombre de runs pour la mesure
        warmup: Nombre de runs d'échauffement
        device: Appareil à utiliser (CPU/GPU)
        is_func: Si True, model_or_func est une fonction qui prend des arguments
        **kwargs: Arguments supplémentaires à passer à la fonction si is_func est True
        
    Returns:
        Débit moyen en exemples par seconde
    """
    if is_func:
        # Si c'est une fonction, on l'appelle directement
        model_func = model_or_func
        batch_size = input_shape[0]
        
        # Échauffement
        for _ in range(warmup):
            _ = model_func(**kwargs)
        
        # Mesure
        start_time = time.time()
        for _ in range(num_runs):
            _ = model_func(**kwargs)
            
    else:
        # Si c'est un modèle PyTorch
        model = model_or_func
        model.eval()
        if device is None:
            device = next(model.parameters()).device
            
        dummy_input = torch.randn(input_shape).to(device)
        batch_size = input_shape[0]
        
        # Échauffement
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Mesure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
    
    total_time = time.time() - start_time
    throughput = (num_runs * batch_size) / total_time
    
    return throughput

def plot_scaling_comparison(results: dict, output_dir: Path = RESULTS_DIR):
    """
    Crée des graphiques comparant le comportement de mise à l'échelle
    
    Args:
        results: Dictionnaire contenant les résultats du benchmark
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Créer le répertoire de sortie si nécessaire
    output_dir.mkdir(exist_ok=True)
    
    # Préparer les données pour les graphiques
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # 1. Temps d'inférence vs Longueur de séquence
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='sequence_length', y='inference_time', 
                 hue='model_name', style='model_name', markers=True, dashes=False)
    plt.title('Temps d\'inférence en fonction de la longueur de séquence')
    plt.xlabel('Longueur de séquence')
    plt.ylabel('Temps d\'inférence (s)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_vs_seq_length.png')
    plt.close()
    
    # 2. Mémoire utilisée vs Longueur de séquence
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='sequence_length', y='memory_usage', 
                 hue='model_name', style='model_name', markers=True, dashes=False)
    plt.title('Utilisation mémoire en fonction de la longueur de séquence')
    plt.xlabel('Longueur de séquence')
    plt.ylabel('Mémoire utilisée (Mo)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_vs_seq_length.png')
    plt.close()
    
    # 3. Débit vs Longueur de séquence
    if 'throughput' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='sequence_length', y='throughput', 
                     hue='model_name', style='model_name', markers=True, dashes=False)
        plt.title('Débit en fonction de la longueur de séquence')
        plt.xlabel('Longueur de séquence')
        plt.ylabel('Débit (exemples/s)')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_vs_seq_length.png')
        plt.close()
    
    # 4. Matrice de corrélation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', linewidths=0.5)
        plt.title('Matrice de corrélation des métriques')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png')
        plt.close()
    
    # 5. Comparaison des FLOPS
    if 'flops' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='model_name', y='flops')
        plt.title('Comparaison des FLOPS')
        plt.xlabel('Modèle')
        plt.ylabel('FLOPS')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'flops_comparison.png')
        plt.close()
    
    # 6. Comparaison du débit
    if 'throughput' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='model_name', y='throughput')
        plt.title('Comparaison du débit')
        plt.xlabel('Modèle')
        plt.ylabel('Exemples par seconde')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_comparison.png')
        plt.close()
    
    # 7. Comparaison de la mémoire
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model_name', y='memory_usage')
    plt.title('Comparaison de l\'utilisation mémoire')
    plt.xlabel('Modèle')
    plt.ylabel('Mémoire utilisée (Mo)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png')
    plt.close()
    
    # 8. Comparaison du temps d'inférence
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model_name', y='inference_time')
    plt.title('Comparaison du temps d\'inférence')
    plt.xlabel('Modèle')
    plt.ylabel('Temps d\'inférence (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png')
    plt.close()
    
    # 9. Comparaison du nombre de paramètres
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model_name', y='parameters')
    plt.title('Comparaison du nombre de paramètres')
    plt.xlabel('Modèle')
    plt.ylabel('Nombre de paramètres')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameters_comparison.png')
    plt.close()

def compare_theoretical_complexity():
    """Compare la complexité théorique des différentes approches"""
    seq_lengths = np.linspace(100, 10000, 100)
    
    # Complexité théorique
    transformer = seq_lengths ** 2  # O(n²) pour l'attention
    linear = seq_lengths  # O(n) pour les approches linéaires
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(seq_lengths, transformer, 'r-', label='Transformer (O(n²))')
    plt.plot(seq_lengths, linear, 'b-', label='Approches linéaires (O(n))')
    
    plt.xlabel('Longueur de séquence (n)')
    plt.ylabel('Complexité computationnelle')
    plt.title('Complexité théorique des approches')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('theoretical_complexity.png')
    plt.close()

@measure_memory_usage
def run_neurolite_agi(input_size: int, seq_length: int, batch_size: int, device: torch.device) -> Tuple[Dict[str, Any], float]:
    """
    Exécute et mesure les performances de NeuroLite avec fonctionnalités AGI.
    
    Args:
        input_size: Taille de la couche d'entrée
        seq_length: Longueur de la séquence d'entrée
        batch_size: Taille du lot
        device: Appareil à utiliser (CPU/GPU)
        
    Returns:
        Tuple contenant (résultats, utilisation_mémoire)
    """
    try:
        # Configuration avec plus de fonctionnalités activées
        config = NeuroLiteConfig.tiny()  # Utiliser une configuration plus petite pour le benchmark
        
        # Activer uniquement les fonctionnalités essentielles pour le benchmark
        config.use_symbolic_module = False
        config.use_bayesian_module = False
        config.use_continual_adapter = False
        config.use_dynamic_routing = False
        config.use_external_memory = False
        config.use_hierarchical_memory = False
        
        # Configurer la projection d'entrée
        config.input_projection_type = "ngram_hash"
        config.vocab_size = 30000
        
        # Ajuster les paramètres pour le benchmark
        config.num_mixer_layers = 4
        config.hidden_size = 128
        config.token_mixing_hidden_size = 256
        config.channel_mixing_hidden_size = 512
        
        # Créer et configurer le modèle
        model = NeuroLiteModel(config).to(device)
        model.eval()
        
        # Créer des données d'entrée simulées
        x = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_length), device=device)
        attention_mask = torch.ones_like(x, device=device)
        
        # Préparer les entrées multimodales
        multimodal_inputs = {
            'input_ids': x,
            'attention_mask': attention_mask
        }
        
        # Mesurer le temps d'inférence
        start_time = time.time()
        with torch.no_grad():
            outputs = model(
                multimodal_inputs=multimodal_inputs,
                use_planning=True,
                update_memory=True
            )
        inference_time = time.time() - start_time
        
        # Gérer différents formats de sortie
        if hasattr(outputs, 'shape'):
            output_shape = tuple(outputs.shape)
        elif isinstance(outputs, dict) and 'logits' in outputs:
            output_shape = tuple(outputs['logits'].shape)
        elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
            output_shape = tuple(outputs['last_hidden_state'].shape)
        else:
            output_shape = str(type(outputs))
        
        # Calculer l'utilisation mémoire après l'exécution
        memory_usage = measure_memory_usage(lambda: None)  # Mesurer la mémoire actuelle
        
        # Créer le dictionnaire de résultats
        result = {
            "model_name": "NeuroLite (AGI)",
            "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "inference_time": inference_time,
            "output_shape": output_shape,
            "sequence_length": seq_length,
            "batch_size": batch_size,
            "flops": 0.0,  # Sera mis à jour plus tard
            "throughput": 0.0,  # Sera mis à jour plus tard
            "memory_usage": memory_usage,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        return result, memory_usage
        
    except Exception as e:
        error_msg = f"Erreur dans run_neurolite_agi: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # Retourner un résultat d'erreur avec les métriques minimales
        error_result = {
            "model_name": "NeuroLite (AGI) - ERREUR",
            "param_count": 0,
            "inference_time": 0.0,
            "output_shape": f"Erreur: {str(e)}",
            "sequence_length": seq_length,
            "batch_size": batch_size,
            "flops": 0.0,
            "throughput": 0.0,
            "memory_usage": 0.0,
            "parameters": 0
        }
        
        return error_result, 0.0

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du dispositif: {device}")
    
    # Créer le répertoire de résultats s'il n'existe pas
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Paramètres de test
    batch_size = 8
    seq_lengths = [64, 128, 256, 512, 1024]  # Différentes longueurs de séquence à tester
    input_size = 768  # Taille d'entrée standard pour les modèles de langue
    
    # Liste pour stocker tous les résultats
    all_results = []
    
    # 1. Comparaison basique des architectures
    print("\n1. Comparaison basique des architectures")
    print("-" * 40)
    
    # Modèles à évaluer
    models_to_test = [
        ("NeuroLite", run_neurolite),
        ("Transformer", run_transformer),
        ("NeuroLite (AGI)", run_neurolite_agi)
    ]
    
    # Modèles pré-entraînés à évaluer
    pretrained_models = [
        "distilbert-base-uncased",
        "google/mobilebert-uncased",
        "microsoft/xtremedistil-l6-h256-uncased"
    ]
    
    # 2. Tests pour différentes longueurs de séquence
    print("\n2. Exécution des benchmarks pour différentes longueurs de séquence")
    print("-" * 40)
    
    for seq_len in seq_lengths:
        print(f"\nLongueur de séquence: {seq_len}")
        print("-" * 30)
        
        # Tester les modèles personnalisés
        for model_name, model_func in models_to_test:
            try:
                print(f"\nTest de {model_name}...")
                
                # Préparer les arguments pour la fonction de modèle
                model_args = {
                    'input_size': input_size,
                    'seq_length': seq_len,
                    'batch_size': batch_size,
                    'device': device
                }
                
                # Exécuter la fonction de modèle
                model_result = model_func(**model_args)
                
                # Gérer différents formats de retour
                if isinstance(model_result, tuple) and len(model_result) == 2:
                    result, memory = model_result
                else:
                    # Si la fonction ne retourne pas de mémoire, on utilise une valeur par défaut
                    result = model_result if isinstance(model_result, dict) else {}
                    memory = 0.0
                
                # S'assurer que result est un dictionnaire
                if not isinstance(result, dict):
                    result = {}
                
                # Ajouter des métriques supplémentaires
                result.update({
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'memory_usage': memory,
                    'parameters': result.get('param_count', 0)
                })
                
                # Mesurer le débit pour les fonctions de modèle
                try:
                    # Préparer les arguments pour la fonction de modèle
                    func_args = {
                        'input_size': input_size,
                        'seq_length': seq_len,
                        'batch_size': batch_size,
                        'device': device
                    }
                    
                    # Créer une fonction partielle avec les bons arguments
                    from functools import partial
                    
                    def run_model_func():
                        try:
                            return model_func(**func_args)
                        except Exception as e:
                            print(f"Erreur lors de l'exécution du modèle pour la mesure du débit: {str(e)}")
                            return None, 0.0
                    
                    throughput = measure_throughput(
                        run_model_func, 
                        input_shape=(batch_size, seq_len, input_size),
                        is_func=True
                    )
                    result['throughput'] = throughput
                except Exception as e:
                    print(f"Erreur lors de la mesure du débit pour {model_name}: {str(e)}")
                    result['throughput'] = 0.0
                
                # Essayer de mesurer les FLOPS si possible
                try:
                    if hasattr(model_func, 'model'):  # Si la fonction a un attribut modèle
                        flops = measure_flops(
                            model_func.model, 
                            input_shape=(batch_size, seq_len, input_size),
                            device=device
                        )
                        result['flops'] = flops
                except Exception as e:
                    print(f"Impossible de mesurer les FLOPS pour {model_name}: {str(e)}")
                    result['flops'] = 0.0
                
                all_results.append(result)
                
                # Afficher les résultats
                print(f"{model_name} - "
                      f"Temps: {result['inference_time']:.4f}s, "
                      f"Paramètres: {result.get('param_count', 'N/A'):,}, "
                      f"Mémoire: {memory:.2f} Mo, "
                      f"Débit: {result.get('throughput', 'N/A'):.2f} ex/s")
                
            except Exception as e:
                print(f"Erreur avec {model_name} (seq_len={seq_len}): {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Tester les modèles pré-entraînés
        for model_name in pretrained_models:
            try:
                print(f"\nTest de {model_name}...")
                
                # Exécuter la fonction de modèle
                result, memory = run_huggingface(model_name, seq_len, batch_size, device)
                
                # Ajouter des métriques de base
                result.update({
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'memory_usage': memory,
                    'parameters': result.get('param_count', 0)
                })
                
                # Essayer de charger le modèle pour les mesures supplémentaires
                try:
                    model = transformers.AutoModel.from_pretrained(model_name).to(device)
                    input_shape = (batch_size, seq_len)
                    
                    # Mesurer le débit
                    try:
                        throughput = measure_throughput(
                            model, 
                            input_shape=input_shape,
                            device=device
                        )
                        result['throughput'] = throughput
                    except Exception as e:
                        print(f"Erreur lors de la mesure du débit pour {model_name}: {str(e)}")
                        result['throughput'] = 0.0
                    
                    # Mesurer les FLOPS si possible
                    try:
                        flops = measure_flops(model, input_shape, device)
                        result['flops'] = flops
                    except Exception as e:
                        print(f"Impossible de mesurer les FLOPS pour {model_name}: {str(e)}")
                        result['flops'] = 0.0
                    
                    # Nettoyer la mémoire
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Impossible de charger le modèle {model_name} pour les mesures avancées: {str(e)}")
                    result['throughput'] = 0.0
                    result['flops'] = 0.0
                
                all_results.append(result)
                
                # Afficher les résultats
                print(f"{model_name} - "
                      f"Temps: {result.get('inference_time', 'N/A'):.4f}s, "
                      f"Paramètres: {result.get('param_count', 0):,}, "
                      f"Mémoire: {memory:.2f} Mo, "
                      f"Débit: {result.get('throughput', 'N/A'):.4f} ex/s")
                
            except Exception as e:
                print(f"Erreur avec {model_name} (seq_len={seq_len}): {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 3. Génération des graphiques et rapports
    print("\n3. Génération des graphiques et rapports")
    print("-" * 40)
    
    try:
        # Générer les graphiques de comparaison
        plot_scaling_comparison(all_results, RESULTS_DIR)
        
        # Générer le rapport détaillé
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generate_benchmark_report(all_results, timestamp)
        
        # Générer le graphique de complexité théorique
        compare_theoretical_complexity()
        
        print(f"\nRapports et graphiques générés dans le répertoire: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"Erreur lors de la génération des rapports: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 4. Affichage des résultats finaux
    print("\n4. Synthèse des performances")
    print("-" * 40)
    
    # Afficher un tableau récapitulatif
    df = pd.DataFrame([
        {
            'Modèle': r.get('model_name', 'N/A'),
            'Longueur': r.get('sequence_length', 'N/A'),
            'Temps (s)': f"{r.get('inference_time', 0):.4f}",
            'Mémoire (Mo)': f"{r.get('memory_usage', 0):.2f}",
            'Paramètres': f"{r.get('parameters', 'N/A'):,}",
            'FLOPS': f"{r.get('flops', 'N/A'):.2e}",
            'Débit (ex/s)': f"{r.get('throughput', 'N/A'):.2f}"
        }
        for r in all_results
    ])
    
    # Afficher le tableau
    print("\nRécapitulatif des performances :")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Sauvegarder les résultats dans un fichier CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'benchmark_results.csv', index=False)
    print(f"\nRésultats détaillés sauvegardés dans : {RESULTS_DIR / 'benchmark_results.csv'}")

if __name__ == "__main__":
    main()
