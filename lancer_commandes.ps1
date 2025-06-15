<#
.SYNOPSIS
    Script PowerShell pour exécuter les commandes principales du projet NeuroLite.

.DESCRIPTION
    Ce script centralise les commandes pour l'entraînement, la génération de texte,
    l'évaluation, et l'inspection du modèle NeuroLite.
    Décommentez la commande que vous souhaitez exécuter.

.NOTES
    Auteur: Gemini
    Assurez-vous d'avoir activé votre environnement virtuel Python avant de lancer ce script.
    Exemple d'activation: .\.venv\Scripts\activate
#>

# --- Configuration Principale ---
# Chemin vers le modèle sauvegardé. Modifiez-le si nécessaire.
$MODEL_PATH = ".\outputs\sft_mixture_of_thoughts"


# --- Commandes Disponibles ---
# Décommentez (retirez le '#') la ligne de la commande que vous souhaitez exécuter.

# 1. Entraîner le modèle (Supervised Fine-Tuning)
#    Utilise les paramètres par défaut. Modifiez les arguments pour personnaliser l'entraînement.
#    Exemple : python scripts/train_sft.py --num_train_epochs 3 --per_device_train_batch_size 4
#
# Write-Host "Lancement de l'entraînement SFT..." -ForegroundColor Green
# python scripts/train_sft.py --max_train_samples 1000


# 2. Générer du texte avec le modèle entraîné
#    N'oubliez pas de modifier le --prompt avec votre phrase de départ.
#
# Write-Host "Lancement de la génération de texte..." -ForegroundColor Green
# python scripts/generate_text.py --model_path $MODEL_PATH --prompt "Explique la théorie de la relativité en termes simples :"


# 3. Évaluer les métriques du modèle sur le jeu de test
#    Calcule la perte (loss) et la perplexité.
#
# Write-Host "Lancement de l'évaluation du modèle..." -ForegroundColor Green
# python scripts/evaluate_model.py --model_path $MODEL_PATH


# 4. Inspecter l'architecture du modèle
#    Affiche l'architecture et le nombre de paramètres d'un modèle avant entraînement.
#
# Write-Host "Lancement de l'inspection du modèle..." -ForegroundColor Green
# python scripts/inspect_model.py


# 5. Tester la passe "forward"
#    Vérifie que les données peuvent circuler dans le modèle sans erreur d'exécution.
#
# Write-Host "Lancement du test de la passe forward..." -ForegroundColor Green
# python scripts/test_forward_pass.py


Write-Host "Script terminé. Aucune commande n'a été exécutée par défaut. Veuillez décommenter une commande pour l'utiliser." -ForegroundColor Yellow 