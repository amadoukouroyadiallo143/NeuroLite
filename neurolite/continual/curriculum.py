"""
Gestionnaire de curriculum pour l'apprentissage continu dans NeuroLite.

Ce module fournit des stratégies pour ordonner les données d'entraînement
de manière à ce que le modèle apprenne des concepts simples avant de passer
à des concepts plus complexes.
"""

from typing import List, Dict, Any, Iterator
import random
import numpy as np

class CurriculumManager:
    """
    Gère dynamiquement le curriculum d'apprentissage en fonction de la performance du modèle.
    """
    def __init__(self, dataset: List[Dict[str, Any]], difficulty_scorer: callable):
        """
        Initialise le gestionnaire de curriculum.

        Args:
            dataset: La liste complète des exemples de données.
            difficulty_scorer: Une fonction qui prend un exemple et retourne un score de difficulté.
        """
        self.dataset = dataset
        self.difficulty_scorer = difficulty_scorer
        print("Scoring du dataset pour le curriculum...")
        self.scored_dataset = self._score_and_normalize_dataset()
        self.sorted_dataset = sorted(self.scored_dataset, key=lambda x: x['difficulty'])
        
        # État dynamique du curriculum
        self.model_competence = 0.0  # Compétence du modèle, de 0.0 (débutant) à 1.0 (expert)
        self.difficulty_window_size = 0.2 # La largeur de la "fenêtre" de difficulté des données à présenter

    def _score_and_normalize_dataset(self) -> List[Dict[str, Any]]:
        """Attribue un score de difficulté et le normalise entre 0 et 1."""
        scored_data = []
        scores = []
        for item in self.dataset:
            difficulty = self.difficulty_scorer(item)
            new_item = item.copy()
            new_item['raw_difficulty'] = difficulty
            scored_data.append(new_item)
            scores.append(difficulty)
        
        # Normaliser les scores
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score: # Éviter la division par zéro
            for item in scored_data:
                item['difficulty'] = 0.5
        else:
            for item in scored_data:
                item['difficulty'] = (item['raw_difficulty'] - min_score) / (max_score - min_score)
        
        return scored_data

    def update_competence(self, metrics: Dict[str, float]):
        """
        Met à jour la compétence du modèle en fonction des métriques de performance.

        Args:
            metrics: Dictionnaire contenant, par exemple, {'eval_loss': 0.5, 'accuracy': 0.8}.
        """
        # Heuristique simple : la compétence augmente si la précision est haute ou la perte est basse.
        # La vitesse d'augmentation dépend de la performance.
        if 'accuracy' in metrics:
            # Si la précision est de 80%, la compétence tend vers 0.8
            self.model_competence = self.model_competence * 0.9 + metrics['accuracy'] * 0.1
        elif 'eval_loss' in metrics:
            # Compétence inversement proportionnelle à la perte (supposons perte max ~2.0)
            competence_from_loss = max(0, 1 - metrics['eval_loss'] / 2.0)
            self.model_competence = self.model_competence * 0.9 + competence_from_loss * 0.1
            
        self.model_competence = np.clip(self.model_competence, 0, 1)
        print(f"Curriculum: Compétence du modèle mise à jour à {self.model_competence:.2f}")

    def get_curriculum_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Fournit un batch de données adapté à la compétence actuelle du modèle.
        """
        # Définir la plage de difficulté cible
        center = self.model_competence
        min_diff = max(0, center - self.difficulty_window_size / 2)
        max_diff = min(1, center + self.difficulty_window_size / 2)

        # Trouver les indices correspondants dans le dataset trié
        start_idx = int(len(self.sorted_dataset) * min_diff)
        end_idx = int(len(self.sorted_dataset) * max_diff)

        if start_idx >= end_idx:
            # Fallback si la fenêtre est vide
            start_idx = max(0, len(self.sorted_dataset) - batch_size)
            end_idx = len(self.sorted_dataset)

        # Échantillonner dans cette sous-section du dataset
        candidate_pool = self.sorted_dataset[start_idx:end_idx]
        
        if not candidate_pool: # Si le pool est vide, prendre depuis le début
             candidate_pool = self.sorted_dataset[:batch_size]

        return random.choices(candidate_pool, k=batch_size)

    def get_curriculum_iterator(self, start_difficulty: float = 0.0, end_difficulty: float = 1.0, pace: str = 'linear') -> Iterator[Dict[str, Any]]:
        """
        Crée un itérateur qui fournit les données selon le curriculum.

        Args:
            start_difficulty: Le niveau de difficulté de départ (0.0 à 1.0).
            end_difficulty: Le niveau de difficulté de fin (0.0 à 1.0).
            pace: La manière de progresser dans la difficulté ('linear', 'sqrt', 'random').

        Yields:
            Un exemple de données à la fois, dans l'ordre du curriculum.
        """
        start_idx = int(len(self.sorted_dataset) * start_difficulty)
        end_idx = int(len(self.sorted_dataset) * end_difficulty)
        
        curriculum_subset = self.sorted_dataset[start_idx:end_idx]

        if pace == 'random':
            random.shuffle(curriculum_subset)
            for item in curriculum_subset:
                yield item
        else: # 'linear' ou autre
            for item in curriculum_subset:
                yield item

# --- Fonctions de scoring de difficulté (exemples) ---

def text_length_scorer(data_item: Dict[str, Any]) -> float:
    """Score la difficulté en fonction de la longueur du texte."""
    if 'text' in data_item:
        return len(data_item['text'].split())
    return 0

def image_resolution_scorer(data_item: Dict[str, Any]) -> float:
    """Score la difficulté en fonction de la résolution de l'image."""
    if 'image' in data_item and hasattr(data_item['image'], 'size'):
        width, height = data_item['image'].size
        return width * height
    return 0

def combined_scorer(data_item: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Combine plusieurs scores de difficulté."""
    score = 0
    if 'text' in data_item:
        score += weights.get('text', 1.0) * text_length_scorer(data_item)
    if 'image' in data_item:
        score += weights.get('image', 1.0) * image_resolution_scorer(data_item)
    return score
