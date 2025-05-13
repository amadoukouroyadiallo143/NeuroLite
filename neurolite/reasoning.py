"""
Module de raisonnement avancé pour NeuroLite.
Implémente des capacités de raisonnement symbolique, planification et inférence
pour des tâches nécessitant un haut niveau d'abstraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

class NeurosymbolicReasoner(nn.Module):
    """
    Module de raisonnement neurosymbolique avancé.
    Combine traitement neuronal et symbolique pour des inférences structurées.
    """
    
    def __init__(
        self,
        hidden_size: int,
        symbolic_dim: int = 64,
        num_inference_steps: int = 3,
        max_facts: int = 100,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.symbolic_dim = symbolic_dim
        self.num_inference_steps = num_inference_steps
        self.max_facts = max_facts
        
        # Extraction d'entités et relations
        self.entity_extractor = nn.Linear(hidden_size, symbolic_dim)
        self.relation_extractor = nn.Linear(hidden_size, symbolic_dim)
        
        # Générateur de règles logiques (triplets SPO)
        self.rule_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, symbolic_dim * 3),  # sujet-prédicat-objet
            nn.Dropout(dropout_rate)
        )
        
        # Moteur d'inférence (transformers légers)
        self.inference_engine = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=symbolic_dim,
                nhead=4,
                dim_feedforward=symbolic_dim * 4,
                dropout=dropout_rate,
                batch_first=True,
                activation='gelu',
                norm_first=True
            ),
            num_layers=2
        )
        
        # Module de vérification de cohérence
        self.consistency_checker = nn.Sequential(
            nn.Linear(symbolic_dim * 2, symbolic_dim),
            nn.LayerNorm(symbolic_dim),
            nn.GELU(),
            nn.Linear(symbolic_dim, 1),
            nn.Sigmoid()
        )
        
        # Projection de sortie pour réintégration
        self.output_projection = nn.Linear(symbolic_dim * 2 + hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        external_facts: Optional[torch.Tensor] = None,
        return_symbolic: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Effectue un raisonnement neurosymbolique sur les représentations.
        
        Args:
            hidden_states: Représentations [batch_size, seq_len, hidden_size]
            external_facts: Faits externes optionnels [batch_size, num_facts, symbolic_dim]
            return_symbolic: Si True, retourne aussi les représentations symboliques
            
        Returns:
            Représentations enrichies [batch_size, seq_len, hidden_size]
            Et optionnellement un dictionnaire de représentations symboliques
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Extraire entités et relations
        entities = self.entity_extractor(hidden_states)
        relations = self.relation_extractor(hidden_states)
        
        # Générer des règles (triplets SPO)
        rules = self.rule_generator(hidden_states)
        rules = rules.view(batch_size, seq_len, 3, self.symbolic_dim)
        
        # Préparer la base de connaissances initiale
        knowledge_base = rules.view(batch_size, seq_len * 3, self.symbolic_dim)
        
        # Intégrer des faits externes si disponibles
        if external_facts is not None:
            # Limiter à max_facts
            if external_facts.size(1) > self.max_facts:
                external_facts = external_facts[:, :self.max_facts, :]
                
            knowledge_base = torch.cat([
                knowledge_base, 
                external_facts
            ], dim=1)
        
        # Combiner les représentations symboliques
        symbolic_repr = torch.cat([entities, relations], dim=-1)
        
        # Effectuer plusieurs étapes d'inférence
        for _ in range(self.num_inference_steps):
            # Préparer l'entrée pour le moteur d'inférence
            # On utilise relations comme requêtes et knowledge_base comme clés/valeurs
            relations_updated = self._inference_step(relations, knowledge_base)
            
            # Vérifier la cohérence des inférences
            consistency_scores = self.consistency_checker(
                torch.cat([relations, relations_updated], dim=-1)
            )
            
            # Mise à jour sélective basée sur la cohérence
            relations = relations * (1 - consistency_scores) + relations_updated * consistency_scores
            
            # Mettre à jour la représentation symbolique
            symbolic_repr = torch.cat([entities, relations], dim=-1)
        
        # Combiner les représentations neuronales et symboliques
        combined = torch.cat([hidden_states, symbolic_repr], dim=-1)
        output = self.output_projection(combined)
        output = self.layer_norm(hidden_states + output)
        
        if return_symbolic:
            symbolic_outputs = {
                'entities': entities,
                'relations': relations,
                'rules': rules,
                'consistency': consistency_scores
            }
            return output, symbolic_outputs
        else:
            return output
            
    def _inference_step(
        self, 
        queries: torch.Tensor, 
        knowledge_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Effectue une étape d'inférence en utilisant le moteur.
        
        Args:
            queries: Représentations de requête [batch_size, seq_len, symbolic_dim]
            knowledge_base: Base de connaissances [batch_size, num_facts, symbolic_dim]
            
        Returns:
            Représentations mises à jour [batch_size, seq_len, symbolic_dim]
        """
        batch_size, seq_len, _ = queries.shape
        
        # Concaténer requêtes et base de connaissances pour l'inférence
        combined_input = torch.cat([queries, knowledge_base], dim=1)
        
        # Créer un masque d'attention pour empêcher l'attention entre requêtes
        # mais permettre l'attention des requêtes vers la base de connaissances
        mask_shape = (batch_size, seq_len + knowledge_base.size(1))
        attn_mask = torch.zeros(mask_shape, device=queries.device)
        
        # Appliquer le transformeur pour propager l'information
        outputs = self.inference_engine(combined_input)
        
        # Extraire uniquement les représentations des requêtes mises à jour
        updated_queries = outputs[:, :seq_len, :]
        
        return updated_queries


class StructuredPlanner(nn.Module):
    """
    Module de planification structurée.
    Génère et évalue des plans d'action pour atteindre des objectifs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_planning_steps: int = 5,
        plan_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_planning_steps = num_planning_steps
        self.plan_dim = plan_dim
        
        # Extracteur d'objectifs
        self.goal_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, plan_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Générateur de plans
        self.plan_generator = nn.GRU(
            input_size=plan_dim,
            hidden_size=plan_dim,
            num_layers=2,
            dropout=dropout_rate if num_planning_steps > 1 else 0,
            batch_first=True
        )
        
        # Évaluateur de plans
        self.plan_evaluator = nn.Sequential(
            nn.Linear(plan_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Module de contraintes (éthiques, sécurité, faisabilité)
        self.constraint_checker = nn.Sequential(
            nn.Linear(plan_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3),  # Différentes contraintes
            nn.Sigmoid()
        )
        
        # Intégration du plan dans la représentation
        self.plan_integration = nn.Linear(hidden_size + plan_dim * num_planning_steps, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
        return_plan: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Génère un plan structuré et l'intègre aux représentations.
        
        Args:
            hidden_states: Représentations [batch_size, seq_len, hidden_size]
            constraints: Contraintes optionnelles [batch_size, 3]
            return_plan: Si True, retourne aussi le plan généré
            
        Returns:
            Représentations enrichies [batch_size, seq_len, hidden_size]
            Et optionnellement un dictionnaire avec le plan et évaluations
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Extraire l'objectif à partir des représentations
        goal_repr = self.goal_extractor(hidden_states.mean(dim=1)).unsqueeze(1)  # [batch, 1, plan_dim]
        
        # Générer un plan étape par étape
        plan_steps = [goal_repr]
        
        # État caché initial pour le GRU
        h_0 = torch.zeros(2, batch_size, self.plan_dim, device=hidden_states.device)
        
        # Générer chaque étape du plan conditionnée sur l'étape précédente
        current_step = goal_repr
        for _ in range(self.num_planning_steps - 1):
            next_step, h_0 = self.plan_generator(current_step, h_0)
            plan_steps.append(next_step)
            current_step = next_step
            
        # Concaténer toutes les étapes du plan
        plan = torch.cat(plan_steps, dim=1)  # [batch, num_steps, plan_dim]
        
        # Évaluer la qualité du plan
        plan_quality = self.plan_evaluator(plan).mean(dim=1)  # [batch, 1]
        
        # Vérifier les contraintes
        plan_constraints = self.constraint_checker(plan)  # [batch, num_steps, 3]
        
        # Appliquer des contraintes externes si fournies
        if constraints is not None:
            # Contraintes externes [batch, 3] -> [batch, 1, 3]
            constraints = constraints.unsqueeze(1)
            plan_constraints = plan_constraints * constraints
            
        # Déterminer si le plan est valide (toutes contraintes satisfaites)
        plan_valid = (plan_constraints.min(dim=2)[0].min(dim=1)[0] > 0.5).float().unsqueeze(1)
        
        # Aplatir le plan pour l'intégration
        flat_plan = plan.view(batch_size, -1)
        
        # Combiner le plan avec les représentations uniquement s'il est valide
        expanded_plan = flat_plan.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([hidden_states, expanded_plan], dim=-1)
        
        # Intégrer le plan dans la représentation
        output = self.plan_integration(combined)
        output = self.layer_norm(hidden_states + output * plan_valid)
        
        if return_plan:
            plan_outputs = {
                'plan': plan,
                'quality': plan_quality,
                'constraints': plan_constraints,
                'valid': plan_valid
            }
            return output, plan_outputs
        else:
            return output
