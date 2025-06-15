"""
Module d'encodage d'objectifs pour NeuroLite.
Permet au modèle de prédire ses propres objectifs et intentions sur différents horizons temporels.
Facilite la planification à long terme et l'adaptabilité aux changements de contexte.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union

class ObjectiveEncoder(nn.Module):
    """
    Module encodant et prédisant les objectifs futurs du modèle.
    Permet une meilleure planification à long terme et adaptabilité contextuelle.
    """
    
    def __init__(
        self,
        hidden_size: int,
        objective_dim: int = 128,
        num_horizons: int = 3,  # Court, moyen et long terme
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.objective_dim = objective_dim
        self.num_horizons = num_horizons
        
        # Extracteur d'objectifs à partir des représentations latentes
        self.objective_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, objective_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Prédicteurs d'objectifs pour différents horizons temporels
        self.horizon_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + objective_dim, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, objective_dim),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_horizons)
        ])
        
        # Module de cohérence entre horizons différents
        self.coherence_checker = nn.Sequential(
            nn.Linear(objective_dim * num_horizons, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_horizons),
            nn.Sigmoid()
        )
        
        # Intégration des objectifs dans la représentation latente
        self.objective_integration = nn.Linear(hidden_size + objective_dim * num_horizons, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Modèle prédictif de l'utilité future des actions
        self.utility_predictor = nn.Sequential(
            nn.Linear(hidden_size + objective_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Valeur d'utilité normalisée entre -1 et 1
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        past_objectives: Optional[torch.Tensor] = None,
        return_objectives: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Encode les objectifs actuels et prédit les objectifs futurs.
        
        Args:
            hidden_states: Représentations [batch_size, seq_len, hidden_size]
            past_objectives: Objectifs historiques optionnels [batch_size, num_horizons, objective_dim]
            return_objectives: Si True, retourne aussi les objectifs encodés
            
        Returns:
            Représentations enrichies [batch_size, seq_len, hidden_size]
            Et optionnellement un dictionnaire avec les objectifs encodés
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Extraire l'objectif actuel à partir des représentations
        current_objective = self.objective_extractor(hidden_states.mean(dim=1))  # [batch, objective_dim]
        
        # Si des objectifs passés sont fournis, les utiliser pour améliorer la prédiction
        if past_objectives is not None:
            # Combiner avec l'objectif actuel par moyenne pondérée
            alpha = 0.8  # Paramètre de pondération entre passé et présent
            current_objective = alpha * current_objective + (1 - alpha) * past_objectives[:, 0, :]
        
        # Prédire les objectifs pour différents horizons temporels
        predicted_objectives = []
        for i in range(self.num_horizons):
            # Pour chaque horizon, prendre en compte l'objectif actuel et le contexte
            context = torch.cat([
                hidden_states.mean(dim=1),  # Représentation globale moyenne
                current_objective  # Objectif actuel
            ], dim=-1)
            
            # Prédire l'objectif futur pour cet horizon
            horizon_objective = self.horizon_predictors[i](context)
            predicted_objectives.append(horizon_objective)
        
        # Concaténer tous les objectifs prédits
        all_objectives = torch.stack(predicted_objectives, dim=1)  # [batch, num_horizons, objective_dim]
        
        # Vérifier la cohérence entre les différents horizons
        flat_objectives = all_objectives.view(batch_size, -1)  # [batch, num_horizons * objective_dim]
        coherence_scores = self.coherence_checker(flat_objectives)  # [batch, num_horizons]
        
        # Ajuster les objectifs en fonction de leur cohérence
        adjusted_objectives = all_objectives * coherence_scores.unsqueeze(-1)
        
        # Intégrer les objectifs dans la représentation pour chaque position de séquence
        objectives_expanded = flat_objectives.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, num_horizons * objective_dim]
        combined = torch.cat([hidden_states, objectives_expanded], dim=-1)
        output = self.objective_integration(combined)
        output = self.layer_norm(hidden_states + output)
        
        if return_objectives:
            # Calculer l'utilité prédite des actions actuelles par rapport aux objectifs
            utility_context = torch.cat([
                hidden_states.mean(dim=1),  # Contexte global
                current_objective  # Objectif actuel
            ], dim=-1)
            utility = self.utility_predictor(utility_context)
            
            objectives_output = {
                'current': current_objective,
                'predicted': all_objectives,
                'coherence': coherence_scores,
                'utility': utility
            }
            return output, objectives_output
        else:
            return output
    
    def compute_objective_loss(
        self,
        current_objectives: torch.Tensor,
        future_objectives: torch.Tensor,
        predicted_objectives: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte pour l'apprentissage de la prédiction d'objectifs.
        
        Args:
            current_objectives: Objectifs actuels [batch, objective_dim]
            future_objectives: Objectifs réels observés dans le futur [batch, num_horizons, objective_dim]
            predicted_objectives: Objectifs prédits [batch, num_horizons, objective_dim]
            
        Returns:
            Perte combinée pour l'apprentissage de la prédiction d'objectifs
        """
        # Perte MSE pour la prédiction d'objectifs futurs
        prediction_loss = F.mse_loss(predicted_objectives, future_objectives)
        
        # Perte de cohérence temporelle (les objectifs successifs devraient être liés)
        coherence_loss = 0.0
        for i in range(1, self.num_horizons):
            coherence_loss += F.cosine_similarity(
                predicted_objectives[:, i, :], 
                predicted_objectives[:, i-1, :],
                dim=1
            ).mean()
        coherence_loss = -coherence_loss / (self.num_horizons - 1)  # Négatif car on maximise la similarité
        
        # Perte combinée
        total_loss = prediction_loss + 0.5 * coherence_loss
        
        return total_loss
