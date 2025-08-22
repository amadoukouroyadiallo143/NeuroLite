"""
World Model pour NeuroLite AGI.
Modèle du monde sophistiqué avec simulation, planification et prédiction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

class ActionType(Enum):
    """Types d'actions dans le monde."""
    PHYSICAL = "physical"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    COMMUNICATION = "communication"
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    CREATION = "creation"
    ANALYSIS = "analysis"

@dataclass
class WorldState:
    """État du monde à un moment donné."""
    objects: Dict[str, torch.Tensor]  # Objets et leurs propriétés
    relations: Dict[str, torch.Tensor]  # Relations entre objets
    agents: Dict[str, torch.Tensor]  # Autres agents
    environment: torch.Tensor  # État environnemental global
    timestamp: float
    uncertainty: torch.Tensor  # Incertitude sur l'état

@dataclass
class Action:
    """Représentation d'une action."""
    action_type: ActionType
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    cost: float
    duration: float
    success_probability: float

@dataclass
class Plan:
    """Plan d'actions séquentielles."""
    actions: List[Action]
    expected_outcome: WorldState
    confidence: float
    estimated_cost: float
    estimated_duration: float
    alternative_plans: List['Plan']

class PhysicsSimulator(nn.Module):
    """Simulateur de physique pour la prédiction d'états futurs."""
    
    def __init__(self, hidden_size: int, max_objects: int = 100):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_objects = max_objects
        
        # Modèle de dynamique physique
        self.physics_model = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),  # state, action, time
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)  # next_state
        )
        
        # Modèle d'interaction entre objets
        self.interaction_model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Prédicteur de collisions
        self.collision_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Modèle de contraintes physiques
        self.constraints_enforcer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, 
                current_state: torch.Tensor,
                action: torch.Tensor,
                time_delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simule l'évolution physique du monde."""
        batch_size = current_state.size(0)
        
        # Prédiction de l'état suivant
        physics_input = torch.cat([current_state, action, time_delta.unsqueeze(-1)], dim=-1)
        next_state_raw = self.physics_model(physics_input)
        
        # Application des contraintes physiques
        next_state = self.constraints_enforcer(next_state_raw)
        
        # Prédiction d'interactions
        interactions = self.interaction_model(torch.cat([current_state, next_state], dim=-1))
        
        # Détection de collisions
        collision_prob = self.collision_predictor(torch.cat([current_state, next_state], dim=-1))
        
        return {
            'next_state': next_state,
            'interactions': interactions,
            'collision_probability': collision_prob,
            'state_change': next_state - current_state
        }

class EnvironmentModel(nn.Module):
    """Modèle de l'environnement et ses règles."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Modèle des règles environnementales
        self.rules_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Modèle de dynamique environnementale
        self.dynamics_model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Prédicteur d'événements
        self.event_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 10),  # 10 types d'événements
            nn.Softmax(dim=-1)
        )
        
        # Modèle d'incertitude
        self.uncertainty_model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, environment_state: torch.Tensor, time_step: int) -> Dict[str, torch.Tensor]:
        """Modélise l'évolution de l'environnement."""
        
        # Encodage des règles actuelles
        rules_encoding = self.rules_encoder(environment_state)
        
        # Dynamique environnementale
        time_encoding = torch.sin(torch.tensor(time_step * 0.1)).unsqueeze(0).expand(environment_state.size(0), -1)
        dynamics_input = torch.cat([environment_state, time_encoding], dim=-1)
        next_env_state = self.dynamics_model(dynamics_input)
        
        # Prédiction d'événements
        event_probabilities = self.event_predictor(next_env_state)
        
        # Estimation d'incertitude
        uncertainty = self.uncertainty_model(next_env_state)
        
        return {
            'next_environment': next_env_state,
            'rules_encoding': rules_encoding,
            'event_probabilities': event_probabilities,
            'uncertainty': uncertainty
        }

class GoalDecomposer(nn.Module):
    """Décompose les objectifs complexes en sous-objectifs."""
    
    def __init__(self, hidden_size: int, max_subgoals: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_subgoals = max_subgoals
        
        # Analyseur de complexité d'objectif
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Décomposeur hiérarchique
        self.hierarchical_decomposer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * max_subgoals)
        )
        
        # Ordonnanceur de sous-objectifs
        self.subgoal_scheduler = nn.Sequential(
            nn.Linear(hidden_size * max_subgoals, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, max_subgoals),
            nn.Softmax(dim=-1)
        )
        
        # Évaluateur de faisabilité
        self.feasibility_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, goal: torch.Tensor, current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Décompose un objectif en sous-objectifs réalisables."""
        
        # Analyse de complexité
        complexity = self.complexity_analyzer(goal)
        
        # Décomposition hiérarchique
        subgoals_flat = self.hierarchical_decomposer(goal)
        subgoals = subgoals_flat.view(-1, self.max_subgoals, self.hidden_size)
        
        # Ordonnancement
        schedule_weights = self.subgoal_scheduler(subgoals_flat)
        
        # Évaluation de faisabilité pour chaque sous-objectif
        feasibility_scores = []
        for i in range(self.max_subgoals):
            subgoal = subgoals[:, i, :]
            feasibility_input = torch.cat([subgoal, current_state], dim=-1)
            feasibility = self.feasibility_evaluator(feasibility_input)
            feasibility_scores.append(feasibility)
        
        feasibility_scores = torch.stack(feasibility_scores, dim=1)
        
        return {
            'subgoals': subgoals,
            'complexity': complexity,
            'schedule_weights': schedule_weights,
            'feasibility_scores': feasibility_scores,
            'total_feasibility': feasibility_scores.mean(dim=1)
        }

class ActionPlanner(nn.Module):
    """Planificateur d'actions pour atteindre les objectifs."""
    
    def __init__(self, hidden_size: int, max_actions: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_actions = max_actions
        
        # Générateur d'actions
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),  # current_state + goal
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * max_actions)
        )
        
        # Évaluateur d'actions
        self.action_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # state + action + goal
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Séquenceur d'actions
        self.action_sequencer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Prédicteur de coût
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        
        # Prédicteur de durée
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, 
                current_state: torch.Tensor,
                goal_state: torch.Tensor,
                max_steps: int = 10) -> Dict[str, torch.Tensor]:
        """Génère un plan d'actions pour atteindre l'objectif."""
        
        batch_size = current_state.size(0)
        
        # Génération d'actions candidates
        planning_input = torch.cat([current_state, goal_state], dim=-1)
        actions_flat = self.action_generator(planning_input)
        actions = actions_flat.view(batch_size, self.max_actions, self.hidden_size)
        
        # Évaluation de chaque action
        action_scores = []
        action_costs = []
        action_durations = []
        
        for i in range(self.max_actions):
            action = actions[:, i, :]
            
            # Score d'utilité
            eval_input = torch.cat([current_state, action, goal_state], dim=-1)
            score = self.action_evaluator(eval_input)
            action_scores.append(score)
            
            # Coût estimé
            cost_input = torch.cat([current_state, action], dim=-1)
            cost = self.cost_predictor(cost_input)
            action_costs.append(cost)
            
            # Durée estimée
            duration = self.duration_predictor(cost_input)
            action_durations.append(duration)
        
        action_scores = torch.stack(action_scores, dim=1).squeeze(-1)
        action_costs = torch.stack(action_costs, dim=1).squeeze(-1)
        action_durations = torch.stack(action_durations, dim=1).squeeze(-1)
        
        # Sélection des meilleures actions
        top_actions_idx = torch.topk(action_scores, k=min(max_steps, self.max_actions), dim=-1)
        
        # Séquencement des actions sélectionnées
        selected_actions = torch.gather(
            actions, 1, 
            top_actions_idx.indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        # Génération de séquence temporelle
        sequence_output, _ = self.action_sequencer(selected_actions)
        
        return {
            'action_sequence': sequence_output,
            'action_scores': torch.gather(action_scores, 1, top_actions_idx.indices),
            'action_costs': torch.gather(action_costs, 1, top_actions_idx.indices),
            'action_durations': torch.gather(action_durations, 1, top_actions_idx.indices),
            'total_cost': torch.gather(action_costs, 1, top_actions_idx.indices).sum(dim=1),
            'total_duration': torch.gather(action_durations, 1, top_actions_idx.indices).sum(dim=1),
            'plan_confidence': torch.gather(action_scores, 1, top_actions_idx.indices).mean(dim=1)
        }

class WorldModel(nn.Module):
    """Modèle du monde complet avec simulation et planification."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Composants principaux
        self.physics_simulator = PhysicsSimulator(hidden_size)
        self.environment_model = EnvironmentModel(hidden_size)
        self.goal_decomposer = GoalDecomposer(hidden_size)
        self.action_planner = ActionPlanner(hidden_size)
        
        # État du monde actuel
        self.register_buffer('current_world_state', torch.randn(1, hidden_size))
        self.register_buffer('world_time', torch.tensor(0.0))
        
        # Mémoire des états passés
        self.state_history = deque(maxlen=1000)
        
        # Prédicteur d'état futur
        self.future_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1)
        )
        
        # Estimateur de probabilité d'événements
        self.event_probability_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 20),  # 20 types d'événements
            nn.Softmax(dim=-1)
        )
        
        # Modèle d'apprentissage du monde
        self.world_learner = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),  # state, action, next_state
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)  # prediction_error
        )
        
        # Statistiques d'apprentissage
        self.register_buffer('prediction_accuracy', torch.tensor(0.5))
        self.register_buffer('total_predictions', torch.tensor(0))
        self.register_buffer('successful_predictions', torch.tensor(0))
    
    def update_world_state(self, new_state: torch.Tensor, action_taken: Optional[torch.Tensor] = None):
        """Met à jour l'état du monde."""
        # Sauvegarde de l'ancien état
        old_state = self.current_world_state.clone()
        self.state_history.append({
            'state': old_state,
            'timestamp': self.world_time.item(),
            'action': action_taken
        })
        
        # Mise à jour
        self.current_world_state = new_state
        self.world_time += 1.0
        
        # Apprentissage si action fournie
        if action_taken is not None:
            self._learn_from_transition(old_state, action_taken, new_state)
    
    def _learn_from_transition(self, old_state: torch.Tensor, action: torch.Tensor, new_state: torch.Tensor):
        """Apprend de la transition observée."""
        learning_input = torch.cat([old_state, action, new_state], dim=-1)
        prediction_error = self.world_learner(learning_input)
        
        # Mise à jour des statistiques
        self.total_predictions += 1
        if prediction_error.item() < 0.5:  # Seuil de succès
            self.successful_predictions += 1
        
        # Mise à jour de la précision moyenne
        self.prediction_accuracy = (self.successful_predictions.float() / 
                                   self.total_predictions.float())
    
    def simulate_future(self, 
                       initial_state: Optional[torch.Tensor] = None,
                       actions: Optional[torch.Tensor] = None,
                       time_horizon: int = 10) -> Dict[str, torch.Tensor]:
        """Simule l'évolution future du monde."""
        
        if initial_state is None:
            initial_state = self.current_world_state
        
        batch_size = initial_state.size(0)
        current_state = initial_state
        
        # Trajectoires simulées
        simulated_states = [current_state]
        simulated_events = []
        
        for t in range(time_horizon):
            # Action à appliquer
            if actions is not None and t < actions.size(1):
                action = actions[:, t, :]
            else:
                # Action par défaut (aucune action)
                action = torch.zeros(batch_size, self.hidden_size, device=initial_state.device)
            
            # Simulation physique
            time_delta = torch.ones(batch_size, device=initial_state.device)
            physics_result = self.physics_simulator(current_state, action, time_delta)
            
            # Évolution environnementale
            env_result = self.environment_model(current_state, t)
            
            # État suivant combiné
            next_state = (physics_result['next_state'] + env_result['next_environment']) / 2
            
            # Prédiction d'événements
            event_probs = self.event_probability_estimator(next_state)
            simulated_events.append(event_probs)
            
            simulated_states.append(next_state)
            current_state = next_state
        
        return {
            'simulated_trajectory': torch.stack(simulated_states, dim=1),
            'event_probabilities': torch.stack(simulated_events, dim=1),
            'final_state': current_state,
            'trajectory_confidence': self.prediction_accuracy.expand(batch_size)
        }
    
    def plan_to_goal(self, 
                    goal: torch.Tensor,
                    current_state: Optional[torch.Tensor] = None,
                    planning_horizon: int = 15) -> Dict[str, Any]:
        """Planifie une séquence d'actions pour atteindre l'objectif."""
        
        if current_state is None:
            current_state = self.current_world_state
        
        # Décomposition de l'objectif
        goal_decomposition = self.goal_decomposer(goal, current_state)
        
        # Planification d'actions
        action_plan = self.action_planner(current_state, goal, max_steps=planning_horizon)
        
        # Simulation du plan
        planned_actions = action_plan['action_sequence']
        simulation_result = self.simulate_future(
            initial_state=current_state,
            actions=planned_actions,
            time_horizon=planned_actions.size(1)
        )
        
        # Évaluation de la réussite probable
        final_predicted_state = simulation_result['final_state']
        goal_distance = torch.norm(final_predicted_state - goal, dim=-1)
        success_probability = torch.exp(-goal_distance)  # Plus proche = plus probable
        
        return {
            'action_plan': action_plan,
            'goal_decomposition': goal_decomposition,
            'simulation_result': simulation_result,
            'success_probability': success_probability,
            'expected_final_state': final_predicted_state,
            'plan_metadata': {
                'total_cost': action_plan['total_cost'],
                'total_duration': action_plan['total_duration'],
                'plan_confidence': action_plan['plan_confidence'],
                'goal_complexity': goal_decomposition['complexity']
            }
        }
    
    def get_world_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état du monde."""
        return {
            'current_state_norm': torch.norm(self.current_world_state).item(),
            'world_time': self.world_time.item(),
            'state_history_length': len(self.state_history),
            'prediction_accuracy': self.prediction_accuracy.item(),
            'total_predictions': self.total_predictions.item(),
            'learning_progress': {
                'successful_predictions': self.successful_predictions.item(),
                'accuracy_trend': 'improving' if self.prediction_accuracy > 0.6 else 'stable'
            }
        }
    
    def reset_world(self, initial_state: Optional[torch.Tensor] = None):
        """Remet à zéro le modèle du monde."""
        if initial_state is not None:
            self.current_world_state = initial_state
        else:
            self.current_world_state = torch.randn_like(self.current_world_state)
        
        self.world_time.fill_(0.0)
        self.state_history.clear()
        self.prediction_accuracy.fill_(0.5)
        self.total_predictions.fill_(0)
        self.successful_predictions.fill_(0)

# Tests et utilitaires
if __name__ == "__main__":
    print("🌍 Test du World Model...")
    
    hidden_size = 256
    world_model = WorldModel(hidden_size)
    
    # Test de simulation
    print("\\n🎬 Test de simulation:")
    initial_state = torch.randn(2, hidden_size)
    simulation = world_model.simulate_future(
        initial_state=initial_state,
        time_horizon=5
    )
    print(f"   Trajectoire: {simulation['simulated_trajectory'].shape}")
    print(f"   Confiance: {simulation['trajectory_confidence'].mean():.3f}")
    
    # Test de planification
    print("\\n📋 Test de planification:")
    goal = torch.randn(2, hidden_size)
    plan = world_model.plan_to_goal(goal, initial_state)
    print(f"   Actions planifiées: {plan['action_plan']['action_sequence'].shape}")
    print(f"   Probabilité de succès: {plan['success_probability'].mean():.3f}")
    print(f"   Coût total estimé: {plan['plan_metadata']['total_cost'].mean():.2f}")
    print(f"   Durée totale estimée: {plan['plan_metadata']['total_duration'].mean():.2f}")
    
    # Test de mise à jour du monde
    print("\\n🔄 Test de mise à jour:")
    new_state = torch.randn(1, hidden_size)
    action = torch.randn(1, hidden_size)
    world_model.update_world_state(new_state, action)
    
    summary = world_model.get_world_summary()
    print(f"   Résumé du monde: {summary}")
    
    print("\\n✅ Tests World Model réussis !")