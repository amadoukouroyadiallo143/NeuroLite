"""
Script pour charger un modèle NeuroLite entraîné et générer du texte.
"""

import os
import sys
import torch
import argparse
import logging
from tqdm import tqdm

# Ajouter le répertoire parent au chemin pour importer NeuroLite
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from training.data_manager import TextTokenizer, DataManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, config, tokenizer, device):
    """Charge un modèle NeuroLite entraîné"""
    logger.info(f"Chargement du modèle depuis {model_path}")
    
    # Charger le checkpoint complet
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraire les informations du checkpoint
    if "model_state_dict" in checkpoint:
        # Format standard avec dictionnaire complet
        model_state = checkpoint["model_state_dict"]
        
        # Vérifier si le vocabulaire est sauvegardé
        if "word_to_idx" in checkpoint and "idx_to_word" in checkpoint:
            logger.info("Utilisation du vocabulaire sauvegardé dans le checkpoint")
            tokenizer.word_to_idx = checkpoint["word_to_idx"]
            tokenizer.idx_to_word = checkpoint["idx_to_word"]
            tokenizer.vocab_initialized = True
            logger.info(f"Vocabulaire chargé avec {len(tokenizer.word_to_idx)} tokens")
        
        # Vérifier si la configuration est sauvegardée
        if "config" in checkpoint:
            logger.info("Utilisation de la configuration sauvegardée dans le checkpoint")
            saved_config = checkpoint["config"]
            
            # Si la configuration est un dictionnaire, convertir en objet NeuroLiteConfig
            if isinstance(saved_config, dict):
                # Créer un nouvel objet de configuration
                config_obj = NeuroLiteConfig()
                
                # Copier les attributs du dictionnaire vers l'objet de configuration
                for key, value in saved_config.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                
                config = config_obj
                logger.info(f"Configuration convertie depuis le dictionnaire sauvegardé")
            else:
                config = saved_config
        
        # Informations complémentaires
        if "epoch" in checkpoint:
            logger.info(f"Modèle entraîné pendant {checkpoint['epoch']} époques")
        if "val_loss" in checkpoint:
            logger.info(f"Perte de validation finale: {checkpoint['val_loss']:.4f}")
    else:
        # Format simple (uniquement state_dict)
        model_state = checkpoint
    
    # Créer le modèle avec la configuration appropriée
    model = NeuroLiteModel(config=config, task_type="generation", tokenizer=tokenizer)
    
    # Nettoyer le state dict pour enlever les clés qui ne sont plus utilisées dans le modèle actuel
    # Pour éviter les erreurs de clés inattendues
    known_keys = set()
    for name, _ in model.named_parameters():
        known_keys.add(name)
    
    cleaned_state = {}
    unexpected_keys = []
    for key, value in model_state.items():
        if key in known_keys:
            cleaned_state[key] = value
        else:
            unexpected_keys.append(key)
    
    # Charger les poids nettoyés (toujours avec strict=False au cas où il manquerait des clés)
    model.load_state_dict(cleaned_state, strict=False)
    
    if unexpected_keys:
        logger.info(f"Information: {len(unexpected_keys)} clés inattendues ont été ignorées lors du chargement")
        if len(unexpected_keys) < 10:  # Afficher uniquement si peu nombreuses
            logger.debug(f"Clés ignorées: {unexpected_keys}")
    
    logger.info("Modèle chargé avec succès")
    
    model.to(device)
    model.eval()
    
    # Afficher la taille du modèle
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Taille du modèle: {num_params:,} paramètres")
    
    return model

def generate_samples(model, tokenizer, device, num_samples=3, prompt="", max_length=100, temperature=0.7):
    """Génère plusieurs exemples de texte"""
    logger.info(f"Génération de {num_samples} exemples de texte avec température {temperature}")
    
    for i in range(num_samples):
        # Si plusieurs échantillons, on peut varier légèrement la température
        current_temp = temperature * (1.0 + (i * 0.05)) if num_samples > 1 else temperature
        
        # Encoder le prompt si fourni, sinon commencer par le token BOS
        if prompt:
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        else:
            input_ids = torch.tensor([[tokenizer.special_tokens['<BOS>']]], dtype=torch.long).to(device)
        
        # Générer le texte
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=current_temp,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.special_tokens.get('<EOS>', None)
        )
        
        # Décoder le texte généré
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Afficher le texte généré
        logger.info(f"Exemple {i+1} (température={current_temp:.2f}):")
        print(f"{generated_text}\n")
        print("-" * 80)

def main(args):
    """Fonction principale"""
    # Déterminer le device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Utilisation du device: {device}")
    
    # Charger le tokenizer
    if args.vocab_file:
        # Charger le vocabulaire depuis un fichier
        tokenizer = TextTokenizer(vocab_size=args.vocab_size)
        # TODO: Implémentation du chargement de vocabulaire
        logger.info(f"Chargement du vocabulaire depuis {args.vocab_file}")
    else:
        # Créer un DataManager pour initialiser le tokenizer
        data_manager = DataManager(
            data_dir=args.data_dir,
            batch_size=32,
            seq_length=args.max_seq_length,  # Utiliser seq_length au lieu de max_seq_length
            vocab_size=args.vocab_size,
            max_samples=100  # Limiter pour ne pas charger trop de données
        )
        tokenizer = data_manager.tokenizer
        logger.info(f"Tokenizer initialisé avec {len(tokenizer.word_to_idx)} tokens")
    
    # Créer la configuration du modèle
    config = NeuroLiteConfig(
        hidden_size=args.hidden_size,
        num_mixer_layers=args.num_layers,  # num_mixer_layers au lieu de num_layers
        token_mixing_hidden_size=args.latent_size,  # token_mixing_hidden_size au lieu de latent_size
        dropout_rate=0.1,  # dropout_rate au lieu de dropout
        vocab_size=args.vocab_size,  # vocab_size au lieu de input_size
        memory_size=args.memory_size,
        memory_dim=args.memory_cell_size,  # memory_dim au lieu de memory_cell_size
        use_external_memory=args.use_memory,  # use_external_memory au lieu de use_memory
        max_seq_length=args.max_seq_length,
        input_projection_type="tokenized_minhash"
    )
    
    # Charger le modèle
    model = load_model(args.model_path, config, tokenizer, device)
    
    # Générer des exemples
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer du texte avec NeuroLite")
    
    # Paramètres du modèle
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Chemin vers le modèle entraîné")
    parser.add_argument("--data_dir", type=str, default="data/wikitext",
                        help="Répertoire contenant les données (pour charger le vocabulaire)")
    parser.add_argument("--vocab_file", type=str, default="",
                        help="Chemin vers le fichier de vocabulaire (optionnel)")
    
    # Paramètres de configuration du modèle
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Taille des couches cachées")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Nombre de couches mixer (num_mixer_layers)")
    parser.add_argument("--latent_size", type=int, default=512,
                        help="Taille du mixing des tokens (token_mixing_hidden_size)")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Taille du vocabulaire")
    parser.add_argument("--memory_size", type=int, default=64,
                        help="Nombre de slots de mémoire")
    parser.add_argument("--use_memory", action="store_true",
                        help="Utiliser la mémoire externe (use_external_memory)")
    parser.add_argument("--memory_cell_size", type=int, default=256,
                        help="Dimension de chaque slot de mémoire (memory_dim)")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Longueur maximale des séquences")
    
    # Paramètres de génération
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Nombre d'exemples à générer")
    parser.add_argument("--prompt", type=str, default="",
                        help="Texte d'amorce pour la génération")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Longueur maximale du texte généré")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Température pour l'échantillonnage (valeurs plus élevées = plus de diversité)")
    parser.add_argument("--cpu", action="store_true",
                        help="Forcer l'utilisation du CPU même si CUDA est disponible")
    
    args = parser.parse_args()
    
    main(args)
