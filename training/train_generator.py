"""
Script d'entraînement pour NeuroLite avec les données Wikitext pour la génération de texte.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
import sys
from tqdm import tqdm

# Ajouter le répertoire parent au chemin pour importer NeuroLite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig

from data_manager import DataManager, TextTokenizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




def train_epoch(model, dataloader, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Boucle d'entraînement
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Déplacer les données sur le device approprié
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Passage avant
        try:
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs["loss"]
            
            # Rétropropagation
            loss.backward()
            
            # Mettre à jour les poids
            optimizer.step()
            
            # Accumuler la perte
            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            logger.error(f"Erreur lors du passage avant pour batch {batch_idx}: {e}")
            logger.error(f"input_ids shape: {input_ids.shape}, target_ids shape: {target_ids.shape}")
            continue  # Passer au batch suivant
    
    # Calculer la perte moyenne sur l'époque (sur les batchs traités avec succès)
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def evaluate(model, dataloader, device):
    """Évalue le modèle sur un ensemble de données"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Déplacer les données sur le device approprié
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Passage avant
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs["loss"]
            
            # Accumuler la perte
            total_loss += loss.item()
            total_batches += 1
    
    # Calculer la perte moyenne en utilisant le nombre réel de batchs traités
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    return avg_loss


def count_parameters(model):
    """Compte le nombre de paramètres entrainables dans le modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sample_text(model, tokenizer, device, prompt="", max_length=50, temperature=0.7):
    """Génère un exemple de texte pour montrer les capacités du modèle"""
    model.eval()
    
    # Encoder le prompt si fourni, sinon commencer par le token BOS
    if prompt:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    else:
        input_ids = torch.tensor([[tokenizer.special_tokens['<BOS>']]], dtype=torch.long).to(device)
    
    # Générer le texte avec la méthode generate du modèle
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        eos_token_id=tokenizer.special_tokens.get('<EOS>', None)
    )
    
    # Décoder le texte généré
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text


def main(args):
    # Configurer le device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Utilisation du device: {device}")
    
    # Créer le gestionnaire de données
    data_manager = DataManager(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        stride=args.stride,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples
    )
    
    # Récupérer les dataloaders (qui sont dans un dictionnaire)
    dataloaders = data_manager.get_dataloaders()
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    test_loader = dataloaders.get("test")
    
    if not train_loader or len(train_loader) == 0:
        logger.error("Impossible de charger les données d'entraînement")
        return
        
    # Vérifier si les jeux de validation et de test sont vides
    if not val_loader or len(val_loader) == 0:
        logger.warning("Aucun échantillon dans le jeu de validation. Utilisation d'une fraction du jeu d'entraînement.")
        # Créer un jeu de validation à partir du jeu d'entraînement
        dataset = train_loader.dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        if val_size > 0:
            # Fonction collate personnalisée pour le dataset
            def collate_fn(batch):
                batch_dict = {}
                for key in batch[0].keys():
                    if key in ["input_ids", "target_ids"]:
                        batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
                    else:
                        batch_dict[key] = [item[key] for item in batch]
                return batch_dict
            
            # Créer les sous-ensembles
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(dataset)))
            
            # Fonction d'accès personnalisée pour les sous-ensembles
            class SubsetWithTransform(torch.utils.data.Subset):
                def __getitem__(self, idx):
                    return self.dataset[self.indices[idx]]
            
            train_dataset = SubsetWithTransform(dataset, train_indices)
            val_dataset = SubsetWithTransform(dataset, val_indices)
            
            # Créer de nouveaux DataLoaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers, 
                collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers, 
                collate_fn=collate_fn
            )
            
            logger.info(f"Jeu de validation créé avec {len(val_dataset)} échantillons")
        else:
            # Si la répartition n'est pas possible, utiliser le jeu d'entraînement pour la validation
            val_loader = train_loader
            logger.warning("Le jeu d'entraînement est trop petit, il sera utilisé aussi pour la validation")
    
    # Si le jeu de test est vide, utiliser le jeu de validation
    if not test_loader or len(test_loader) == 0:
        logger.warning("Aucun échantillon dans le jeu de test. Utilisation du jeu de validation.")
        test_loader = val_loader
    
    # Configurer le modèle
    config = NeuroLiteConfig(
        hidden_size=args.hidden_size,
        num_mixer_layers=args.num_layers,
        max_seq_length=args.seq_length,
        use_external_memory=args.use_memory,
        memory_size=args.memory_size,
        dropout_rate=args.dropout_rate,
        vocab_size=args.vocab_size,
        input_projection_type="tokenized_minhash"
    )
    
    # Créer le modèle unifié avec task_type="generation"
    model = NeuroLiteModel(
        config=config, 
        task_type="generation", 
        tokenizer=data_manager.tokenizer
    )
    model.to(device)
    
    # Afficher la taille du modèle
    num_params = count_parameters(model)
    logger.info(f"Taille du modèle: {num_params:,} paramètres")
    
    # Configurer l'optimiseur (pas besoin de définir criterion car le modèle calcule la perte en interne)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Entraîner le modèle
    logger.info("Début de l'entraînement du modèle de génération de texte")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Entraînement
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch}/{args.num_epochs} - Val Loss: {val_loss:.4f}")
        
        # Générer un exemple de texte pour voir les progrès
        if epoch % args.sample_every == 0:
            sample = sample_text(model, data_manager.tokenizer, device, max_length=50)
            logger.info(f"Exemple de texte généré (epoch {epoch}):\n{sample}\n")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Sauvegarder le modèle
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__,
                # Sauvegarder le vocabulaire pour pouvoir recharger le modèle
                'word_to_idx': data_manager.tokenizer.word_to_idx,
                'idx_to_word': data_manager.tokenizer.idx_to_word,
            }, os.path.join(args.output_dir, "best_generator.pt"))
            logger.info(f"Nouveau meilleur modèle sauvegardé avec Val Loss: {val_loss:.4f}")
    
    # Évaluation finale sur l'ensemble de test
    if test_loader:
        logger.info("Évaluation finale sur l'ensemble de test")
        test_loss = evaluate(model, test_loader, device)
        logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Générer quelques exemples avec le modèle final
    logger.info("Génération d'exemples avec le modèle final:")
    for _ in range(3):
        sample = sample_text(model, data_manager.tokenizer, device, max_length=100)
        logger.info(f"Texte généré:\n{sample}\n")
    
    logger.info("Entraînement terminé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement pour NeuroLite Generator")
    
    # Paramètres des données
    parser.add_argument("--data_dir", type=str, default="data/wikitext", help="Répertoire des données")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des batchs")
    parser.add_argument("--num_workers", type=int, default=4, help="Nombre de workers pour le chargement des données")
    parser.add_argument("--seq_length", type=int, default=128, help="Longueur des séquences d'entraînement")
    parser.add_argument("--stride", type=int, default=64, help="Pas pour la fenêtre glissante")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Taille du vocabulaire")
    parser.add_argument("--max_samples", type=int, default=None, help="Nombre maximum d'échantillons à charger par division (pour les tests)")

    
    # Paramètres du modèle
    parser.add_argument("--hidden_size", type=int, default=256, help="Dimension des représentations cachées")
    parser.add_argument("--num_layers", type=int, default=6, help="Nombre de couches mixer")
    parser.add_argument("--use_memory", action="store_true", help="Utiliser la mémoire externe")
    parser.add_argument("--memory_size", type=int, default=128, help="Taille de la mémoire externe")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Taux de dropout")
    
    # Paramètres d'entraînement
    parser.add_argument("--num_epochs", type=int, default=20, help="Nombre d'époques d'entraînement")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Taux d'apprentissage")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--no_cuda", action="store_true", help="Désactiver l'utilisation de CUDA")
    parser.add_argument("--output_dir", type=str, default="models", help="Répertoire de sortie pour les modèles sauvegardés")
    parser.add_argument("--sample_every", type=int, default=2, help="Générer un exemple tous les n epochs")
    
    args = parser.parse_args()
    
    main(args)