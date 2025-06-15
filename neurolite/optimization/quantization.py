"""
Techniques de quantification pour optimiser les modèles NeuroLite.

Ce module fournit des fonctions pour appliquer la quantification,
réduisant la taille du modèle et accélérant l'inférence.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert, get_default_qconfig
from typing import List, Optional
import os

def apply_dynamic_quantization(model: nn.Module, layer_types: Optional[List[type]] = None):
    """
    Applique la quantification dynamique à un modèle.
    C'est la méthode la plus simple, convertissant les poids en int8 au moment de l'exécution.

    Args:
        model: Le modèle à quantifier.
        layer_types: Les types de couches spécifiques à quantifier (ex: [nn.Linear]).
                     Si None, quantifie uniquement les couches Linear par défaut.
    """
    if layer_types is None:
        layer_types = {nn.Linear}
        
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec=layer_types,
        dtype=torch.qint8
    )
    return quantized_model

def apply_static_quantization(model: nn.Module, calibration_data: List[torch.Tensor], inplace: bool = False):
    """
    Applique la quantification statique post-entraînement.
    Nécessite des données de calibration pour déterminer les paramètres de quantification.

    Args:
        model: Le modèle à quantifier.
        calibration_data: Une liste de tenseurs d'entrée représentatifs pour calibrer le modèle.
        inplace: Si True, modifie le modèle directement.
    
    Returns:
        Le modèle quantifié.
    """
    if not inplace:
        model_to_quantize = model.copy()
    else:
        model_to_quantize = model
        
    model_to_quantize.eval()
    
    # Spécifier la configuration de quantification (par défaut ici)
    qconfig = get_default_qconfig('fbgemm') # 'fbgemm' pour x86, 'qnnpack' pour ARM
    model_to_quantize.qconfig = qconfig
    
    # Préparer le modèle en insérant les observateurs
    prepare(model_to_quantize, inplace=True)
    
    # Calibrer le modèle avec les données fournies
    print("Calibration du modèle pour la quantification statique...")
    with torch.no_grad():
        for data in calibration_data:
            # Assurez-vous que le forward du modèle peut accepter ce type de données
            model_to_quantize(data)
    print("Calibration terminée.")
    
    # Convertir le modèle en sa version quantifiée
    convert(model_to_quantize, inplace=True)
    
    return model_to_quantize

def print_model_size(model: nn.Module):
    """Affiche la taille du modèle en mégaoctets."""
    torch.save(model.state_dict(), "temp_quant.p")
    size_mb = os.path.getsize("temp_quant.p") / 1e6
    os.remove("temp_quant.p")
    print(f"Taille du modèle : {size_mb:.2f} MB")
