"""
Contrôleur AGI Central pour NeuroLite.
Orchestre toutes les capacités cognitives et coordonne les sous-systèmes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

class CognitiveMode(Enum):
    """Modes cognitifs de l'AGI."""
    LEARNING = "learning"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    METACOGNITIVE = "metacognitive"
    PLANNING = "planning"
    REACTIVE = "reactive"

@dataclass
class CognitiveState:
    """État cognitif actuel de l'AGI."""
    mode: CognitiveMode
    attention_focus: torch.Tensor
    working_memory: torch.Tensor
    confidence_level: float
    energy_level: float
    curiosity_level: float
    meta_awareness: float
    goal_clarity: float

@dataclass
class AGITask:
    """Représentation d'une tâche pour l'AGI."""
    task_type: str
    priority: float
    context: Dict[str, Any]
    requirements: List[str]
    expected_output_type: str
    difficulty_estimate: float

class CognitiveOrchestrator(nn.Module):
    """
    Orchestrateur central qui coordonne tous les processus cognitifs.
    """
    
    def __init__(self, hidden_size: int, num_cognitive_modes: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cognitive_modes = num_cognitive_modes
        
        # Analyseur de contexte
        self.context_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Sélecteur de mode cognitif
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_cognitive_modes),
            nn.Softmax(dim=-1)
        )
        
        # Contrôleur d'attention
        self.attention_controller = nn.MultiheadAttention(
            hidden_size, num_heads=16, dropout=0.1, batch_first=True
        )
        
        # Estimateur de confiance
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Moniteur d'énergie cognitive
        self.energy_monitor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Générateur de curiosité
        self.curiosity_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Mémoire de travail
        self.working_memory_capacity = 32
        self.working_memory = nn.Parameter(
            torch.randn(self.working_memory_capacity, hidden_size) * 0.02
        )
        self.working_memory_gate = nn.Linear(hidden_size, self.working_memory_capacity)
        
        # États cognitifs
        self.register_buffer('current_mode', torch.tensor(0))
        self.register_buffer('cognitive_energy', torch.tensor(1.0))
        self.register_buffer('focus_intensity', torch.tensor(0.5))
        
    def forward(self, 
                input_context: torch.Tensor,
                task_specification: Optional[AGITask] = None,
                previous_state: Optional[CognitiveState] = None) -> Tuple[CognitiveState, Dict[str, Any]]:
        """
        Détermine l'état cognitif optimal pour le contexte donné.
        """
        batch_size, seq_len, _ = input_context.shape
        
        # Analyse du contexte
        context_features = self.context_analyzer(input_context)
        context_summary = context_features.mean(dim=1)  # [batch, hidden_size]
        
        # Sélection du mode cognitif
        mode_probs = self.mode_selector(context_summary)
        selected_mode = torch.argmax(mode_probs, dim=-1)
        
        # Contrôle attentionnel
        attention_output, attention_weights = self.attention_controller(
            context_features, context_features, context_features
        )
        attention_focus = attention_output.mean(dim=1)
        
        # Mise à jour de la mémoire de travail
        memory_gate_weights = torch.sigmoid(self.working_memory_gate(context_summary))
        memory_update = torch.einsum('bh,mh->bm', context_summary, self.working_memory)
        memory_update = memory_update * memory_gate_weights
        
        # Mise à jour adaptive de la mémoire de travail
        with torch.no_grad():
            memory_decay = 0.95
            self.working_memory.data *= memory_decay
            update_strength = memory_update.mean(dim=0) * 0.01
            self.working_memory.data += update_strength.unsqueeze(-1) * context_summary.mean(dim=0).unsqueeze(0)
        
        # Estimation des métriques cognitives
        confidence = self.confidence_estimator(attention_focus).squeeze(-1)
        energy = self.energy_monitor(attention_focus).squeeze(-1)
        curiosity = self.curiosity_generator(attention_focus).squeeze(-1)
        
        # Calcul de la méta-conscience
        meta_awareness = self._compute_meta_awareness(context_features, attention_weights)
        
        # Clarté d'objectif
        goal_clarity = self._compute_goal_clarity(task_specification, context_summary)
        
        # Création de l'état cognitif
        cognitive_state = CognitiveState(
            mode=list(CognitiveMode)[selected_mode[0].item()],
            attention_focus=attention_focus,
            working_memory=self.working_memory.clone(),
            confidence_level=confidence.mean().item(),
            energy_level=energy.mean().item(),
            curiosity_level=curiosity.mean().item(),
            meta_awareness=meta_awareness,
            goal_clarity=goal_clarity
        )
        
        # Mise à jour des buffers internes
        self.current_mode = selected_mode[0]
        self.cognitive_energy = self.cognitive_energy * 0.99 + energy.mean() * 0.01
        self.focus_intensity = self.focus_intensity * 0.9 + attention_weights.max() * 0.1
        
        # Métadonnées
        metadata = {
            'mode_probabilities': mode_probs,
            'attention_weights': attention_weights,
            'memory_gate_weights': memory_gate_weights,
            'context_complexity': self._estimate_context_complexity(context_features),
            'cognitive_load': self._estimate_cognitive_load(context_features, attention_weights)
        }
        
        return cognitive_state, metadata
    
    def _compute_meta_awareness(self, context_features: torch.Tensor, attention_weights: torch.Tensor) -> float:
        """Calcule le niveau de méta-conscience."""
        # Diversité attentionnelle comme proxy de méta-conscience
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        return attention_entropy.mean().item()
    
    def _compute_goal_clarity(self, task_spec: Optional[AGITask], context_summary: torch.Tensor) -> float:
        """Calcule la clarté de l'objectif."""
        if task_spec is None:
            return 0.5  # Clarté moyenne sans spécification
        
        # Utilise la difficulté estimée et la priorité
        clarity = (1.0 - task_spec.difficulty_estimate) * task_spec.priority
        return min(1.0, max(0.0, clarity))
    
    def _estimate_context_complexity(self, context_features: torch.Tensor) -> float:
        """Estime la complexité du contexte."""
        # Variance comme mesure de complexité
        complexity = torch.var(context_features, dim=-1).mean().item()
        return min(1.0, complexity / 10.0)  # Normalisation
    
    def _estimate_cognitive_load(self, context_features: torch.Tensor, attention_weights: torch.Tensor) -> float:
        """Estime la charge cognitive actuelle."""
        # Combinaison de la complexité du contexte et de la dispersion attentionnelle
        context_load = torch.norm(context_features, dim=-1).mean()
        attention_load = 1.0 - torch.max(attention_weights, dim=-1)[0].mean()
        total_load = (context_load.item() + attention_load.item()) / 2
        return min(1.0, total_load)

class GoalManager(nn.Module):
    """Gestionnaire d'objectifs hiérarchiques."""
    
    def __init__(self, hidden_size: int, max_goals: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_goals = max_goals
        
        # Représentation des objectifs
        self.goal_embeddings = nn.Parameter(
            torch.randn(max_goals, hidden_size) * 0.02
        )
        
        # Priorité des objectifs
        self.goal_priorities = nn.Parameter(torch.ones(max_goals))
        
        # Réseau de planification
        self.planner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Évaluateur de progression
        self.progress_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # État des objectifs
        self.register_buffer('goal_states', torch.zeros(max_goals))
        self.register_buffer('goal_completion', torch.zeros(max_goals))
        
    def forward(self, current_state: torch.Tensor, desired_outcome: torch.Tensor) -> Dict[str, Any]:
        """Plan et gère les objectifs."""
        batch_size = current_state.size(0)
        
        # Sélection d'objectifs pertinents
        goal_relevance = torch.matmul(current_state.mean(dim=1), self.goal_embeddings.T)
        active_goals = torch.topk(goal_relevance, k=min(4, self.max_goals), dim=-1)
        
        # Planification pour chaque objectif
        plans = []
        for i, goal_idx in enumerate(active_goals.indices[0]):
            goal_emb = self.goal_embeddings[goal_idx]
            combined_input = torch.cat([current_state.mean(dim=1), goal_emb.unsqueeze(0)], dim=-1)
            plan = self.planner(combined_input)
            plans.append(plan)
        
        # Évaluation de la progression
        progress_scores = []
        for plan in plans:
            progress = self.progress_evaluator(plan)
            progress_scores.append(progress)
        
        return {
            'active_goals': active_goals.indices[0].tolist(),
            'goal_plans': plans,
            'progress_scores': progress_scores,
            'goal_priorities': self.goal_priorities[active_goals.indices[0]].tolist()
        }
    
    def update_goal_completion(self, goal_id: int, completion_score: float):
        """Met à jour la completion d'un objectif."""
        if 0 <= goal_id < self.max_goals:
            self.goal_completion[goal_id] = completion_score
            
            # Ajustement adaptatif de la priorité
            if completion_score > 0.9:
                self.goal_priorities.data[goal_id] *= 0.8  # Réduire priorité si complété
            elif completion_score < 0.1:
                self.goal_priorities.data[goal_id] *= 1.1  # Augmenter priorité si difficile

class AGICentralController(nn.Module):
    """Contrôleur central de l'AGI qui orchestre tous les sous-systèmes."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Composants principaux
        self.cognitive_orchestrator = CognitiveOrchestrator(hidden_size)
        self.goal_manager = GoalManager(hidden_size)
        
        # Interface de communication inter-modules
        self.module_router = nn.ModuleDict({
            'memory': nn.Linear(hidden_size, hidden_size),
            'reasoning': nn.Linear(hidden_size, hidden_size),
            'planning': nn.Linear(hidden_size, hidden_size),
            'learning': nn.Linear(hidden_size, hidden_size)
        })
        
        # Coordinateur global
        self.global_coordinator = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Méta-contrôleur
        self.meta_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4),  # 4 stratégies de contrôle
            nn.Softmax(dim=-1)
        )
        
        # Historique des décisions
        self.decision_history = []
        self.max_history = 100
        
    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                task: Optional[AGITask] = None,
                module_states: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Contrôle central de tous les processus AGI."""
        
        # Extraction du contexte principal
        main_context = inputs.get('context', inputs.get('text', list(inputs.values())[0]))
        
        # Analyse cognitive
        cognitive_state, cog_metadata = self.cognitive_orchestrator(main_context, task)
        
        # Gestion des objectifs
        desired_outcome = inputs.get('desired_outcome', cognitive_state.attention_focus)
        goal_info = self.goal_manager(main_context, desired_outcome)
        
        # Routage vers les modules spécialisés
        module_commands = {}
        for module_name, router in self.module_router.items():
            command = router(cognitive_state.attention_focus)
            module_commands[module_name] = command
        
        # Coordination globale
        if module_states:
            all_states = [module_states.get(name, torch.zeros_like(cmd)) 
                         for name, cmd in module_commands.items()]
            global_state = torch.cat(all_states, dim=-1)
            coordinated_state = self.global_coordinator(global_state)
        else:
            coordinated_state = cognitive_state.attention_focus
        
        # Méta-contrôle
        meta_strategy = self.meta_controller(coordinated_state)
        
        # Compilation de la décision
        decision = {
            'cognitive_state': cognitive_state,
            'goal_info': goal_info,
            'module_commands': module_commands,
            'coordinated_state': coordinated_state,
            'meta_strategy': meta_strategy,
            'metadata': {
                'cognitive_metadata': cog_metadata,
                'decision_confidence': cognitive_state.confidence_level,
                'system_energy': cognitive_state.energy_level
            }
        }
        
        # Historique
        self.decision_history.append({
            'mode': cognitive_state.mode.value,
            'confidence': cognitive_state.confidence_level,
            'energy': cognitive_state.energy_level
        })
        
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        return decision
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du système AGI."""
        recent_decisions = self.decision_history[-10:] if self.decision_history else []
        
        avg_confidence = np.mean([d['confidence'] for d in recent_decisions]) if recent_decisions else 0.5
        avg_energy = np.mean([d['energy'] for d in recent_decisions]) if recent_decisions else 0.5
        
        mode_distribution = {}
        for decision in recent_decisions:
            mode = decision['mode']
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        
        return {
            'average_confidence': avg_confidence,
            'average_energy': avg_energy,
            'mode_distribution': mode_distribution,
            'decision_count': len(self.decision_history),
            'current_cognitive_energy': self.cognitive_orchestrator.cognitive_energy.item(),
            'current_focus_intensity': self.cognitive_orchestrator.focus_intensity.item(),
            'active_goals_count': torch.sum(self.goal_manager.goal_completion > 0.1).item()
        }
    
    def reset_system(self):
        """Remet à zéro le système AGI."""
        self.decision_history = []
        self.cognitive_orchestrator.cognitive_energy.fill_(1.0)
        self.cognitive_orchestrator.focus_intensity.fill_(0.5)
        self.goal_manager.goal_completion.fill_(0.0)
        self.goal_manager.goal_priorities.fill_(1.0)
        
        with torch.no_grad():
            self.cognitive_orchestrator.working_memory.normal_(0, 0.02)
            self.goal_manager.goal_embeddings.normal_(0, 0.02)

# Utilitaires pour la création de tâches AGI
def create_agi_task(task_type: str, context: Dict[str, Any], priority: float = 1.0) -> AGITask:
    """Crée une tâche AGI standardisée."""
    difficulty_map = {
        'classification': 0.3,
        'generation': 0.5,
        'reasoning': 0.7,
        'planning': 0.8,
        'creative': 0.6,
        'learning': 0.9
    }
    
    return AGITask(
        task_type=task_type,
        priority=priority,
        context=context,
        requirements=context.get('requirements', []),
        expected_output_type=context.get('output_type', 'text'),
        difficulty_estimate=difficulty_map.get(task_type, 0.5)
    )

# Tests
if __name__ == "__main__":
    print("🧠 Test du Contrôleur AGI Central...")
    
    hidden_size = 256
    batch_size, seq_len = 2, 32
    
    # Initialisation
    agi_controller = AGICentralController(hidden_size)
    
    # Données de test
    test_input = {
        'context': torch.randn(batch_size, seq_len, hidden_size),
        'desired_outcome': torch.randn(batch_size, hidden_size)
    }
    
    # Tâche de test
    test_task = create_agi_task(
        'reasoning',
        {'description': 'Résoudre un problème complexe', 'output_type': 'solution'},
        priority=0.8
    )
    
    # Test forward
    with torch.no_grad():
        decision = agi_controller(test_input, test_task)
        
        print(f"Mode cognitif: {decision['cognitive_state'].mode.value}")
        print(f"Confiance: {decision['cognitive_state'].confidence_level:.3f}")
        print(f"Énergie: {decision['cognitive_state'].energy_level:.3f}")
        print(f"Curiosité: {decision['cognitive_state'].curiosity_level:.3f}")
        print(f"Méta-conscience: {decision['cognitive_state'].meta_awareness:.3f}")
        
        status = agi_controller.get_system_status()
        print(f"\nStatut système: {status}")
        
    print("✅ Tests AGI Controller réussis !")