"""
Script pour télécharger le jeu de données Mixture-of-Thoughts depuis le Hub Hugging Face.
"""
import os
from pathlib import Path
from datasets import load_dataset

def download_and_save_dataset():
    """
    Télécharge le jeu de données 'open-r1/Mixture-of-Thoughts' et le sauvegarde
    dans le répertoire 'data/raw/mixture_of_thoughts' pour être utilisé par les
    scripts d'entraînement.
    """
    dataset_name = "open-r1/Mixture-of-Thoughts"
    save_path = Path("data/raw/mixture_of_thoughts")

    # Vérifie si le jeu de données existe déjà pour éviter de le télécharger à nouveau
    # On vérifie la présence d'un fichier clé que `save_to_disk` crée.
    if (save_path / "dataset_dict.json").exists():
        print(f"Le jeu de données semble déjà exister dans : {save_path}")
        print("Téléchargement ignoré.")
        return

    print(f"Téléchargement du jeu de données '{dataset_name}' depuis le Hub Hugging Face...")
    
    # Crée les répertoires parents si nécessaire
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Charge le jeu de données depuis le Hub, en spécifiant la configuration 'all'
    dataset = load_dataset(dataset_name, name='all')
    
    print(f"Sauvegarde du jeu de données dans : {save_path}...")
    
    # Sauvegarde le jeu de données sur le disque.
    # Cette méthode crée la structure de dossiers attendue par `load_from_disk`.
    dataset.save_to_disk(save_path)
    
    print("Jeu de données téléchargé et sauvegardé avec succès.")
    print(f"Les données d'entraînement sont prêtes à être utilisées dans : {save_path / 'train'}")

if __name__ == "__main__":
    download_and_save_dataset() 