"""
Interface pour l'interaction de NeuroLite avec des environnements externes.

Ce module définit une API standard pour que les agents NeuroLite puissent
interagir avec divers environnements (simulations, jeux, etc.), suivant
une approche similaire à celle de l'API Gym de Farama-Foundation (anciennement OpenAI Gym).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

class EnvironmentInterface(ABC):
    """
    Classe d'interface abstraite pour un environnement interactif.
    """

    def __init__(self):
        self._action_space = None
        self._observation_space = None

    @property
    def action_space(self) -> Any:
        """Retourne l'espace des actions possibles."""
        return self._action_space

    @property
    def observation_space(self) -> Any:
        """Retourne l'espace des observations possibles."""
        return self._observation_space

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.

        Args:
            action: L'action à effectuer.

        Returns:
            Une tuple contenant:
            - observation: L'observation de l'état suivant.
            - reward: La récompense obtenue.
            - done: Un booléen indiquant si l'épisode est terminé.
            - info: Un dictionnaire d'informations de débogage.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Réinitialise l'environnement à un état initial.

        Args:
            seed: Graine optionnelle pour la reproductibilité.

        Returns:
            L'observation initiale.
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> Any:
        """
        Affiche l'état actuel de l'environnement.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Ferme l'environnement et nettoie les ressources.
        """
        pass

class Agent:
    """
    Classe de base pour un agent qui interagit avec un environnement.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Sélectionne une action basée sur l'observation actuelle.
        """
        pass

def run_episode(env: EnvironmentInterface, agent: Agent, max_steps: int = 1000) -> float:
    """
    Exécute un épisode complet d'interaction entre l'agent et l'environnement.

    Args:
        env: L'instance de l'environnement.
        agent: L'agent qui interagit avec l'environnement.
        max_steps: Le nombre maximum de pas dans l'épisode.

    Returns:
        La récompense totale accumulée pendant l'épisode.
    """
    observation = env.reset()
    total_reward = 0.0
    done = False
    
    for step in range(max_steps):
        action = agent.select_action(observation)
        next_observation, reward, done, info = env.step(action)
        
        # Ici, on pourrait ajouter la logique d'apprentissage de l'agent
        # (ex: stocker l'expérience dans un buffer de rejeu)
        
        observation = next_observation
        total_reward += reward
        
        if done:
            break
            
    print(f"Épisode terminé après {step + 1} pas. Récompense totale: {total_reward}")
    return total_reward
