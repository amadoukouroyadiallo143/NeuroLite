import os
from datasets import load_dataset

# Chemin de destination
output_dir = r"c:\Users\Diallo\Dev\ai\NeuroLite\data\raw\mixture_of_thoughts"

# Créer le répertoire s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

def download_mixture_of_thoughts():
    print("Téléchargement du dataset Mixture-of-Thoughts...")
    
    try:
        # Charger le dataset
        dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all")
        
        # Sauvegarder en format parquet
        print("Sauvegarde des données...")
        dataset.save_to_disk(output_dir)
        
        # Afficher les informations du dataset
        print("\nTéléchargement terminé avec succès !")
        print(f"Données enregistrées dans : {output_dir}")
        print("\nAperçu du dataset :")
        print(dataset)
        
        # Afficher les statistiques
        print("\nStatistiques :")
        for split in dataset:
            print(f"\nSplit: {split}")
            print(f"Nombre d'exemples: {len(dataset[split])}")
            if len(dataset[split]) > 0:
                print("\nPremier exemple :")
                print(dataset[split][0])
        
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")

if __name__ == "__main__":
    download_mixture_of_thoughts()