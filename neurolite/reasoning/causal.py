"""
Implémentation des capacités de raisonnement causal pour NeuroLite.

Ce module fournit des outils pour construire et interroger des modèles causaux,
permettant au modèle de raisonner sur les causes et les effets.
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import List, Dict, Optional, Tuple

class CausalGraph(nn.Module):
    """
    Représente un graphe causal structurel (Structural Causal Model - SCM) neuronal.
    Le graphe est défini par un ensemble de variables, les relations de cause à effet
    et des mécanismes fonctionnels (réseaux de neurones) pour chaque variable.
    """
    def __init__(self, variables: List[str], edges: List[Tuple[str, str]], var_dims: Dict[str, int]):
        super().__init__()
        self.variables = sorted(list(variables))
        self.var_to_idx = {var: i for i, var in enumerate(self.variables)}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.variables)
        self.graph.add_edges_from(edges)
        self.var_dims = var_dims

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Le graphe causal doit être un graphe orienté acyclique (DAG).")

        # Créer un mécanisme (réseau de neurones) pour chaque variable
        self.mechanisms = nn.ModuleDict()
        for var in self.variables:
            parents = list(self.graph.predecessors(var))
            # La dimension d'entrée est la somme des dimensions des parents + une dimension pour le bruit
            input_dim = sum(self.var_dims[p] for p in parents) + self.var_dims[var] # Bruit de même dim que var
            output_dim = self.var_dims[var]
            
            # Un simple MLP pour modéliser le mécanisme causal
            self.mechanisms[var] = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )

    def forward(self, 
                noise: Dict[str, torch.Tensor], 
                intervention: Optional[Dict[str, torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Génère des données à partir du modèle causal, en appliquant potentiellement une intervention.

        Args:
            noise: Un dictionnaire de tenseurs de bruit exogène pour chaque variable.
            intervention: Dictionnaire optionnel spécifiant les interventions. 
                          La clé est le nom de la variable, la valeur est le tenseur de la valeur à imposer.

        Returns:
            Un dictionnaire contenant les valeurs générées pour chaque variable.
        """
        if intervention is None:
            intervention = {}
            
        generated_values = {}
        # S'assurer de traiter les variables dans l'ordre topologique
        for var in nx.topological_sort(self.graph):
            # Si la variable est sous intervention, on fixe sa valeur
            if var in intervention:
                generated_values[var] = intervention[var]
                continue

            parents = list(self.graph.predecessors(var))
            
            if not parents:
                # Variable racine, dépend uniquement du bruit
                parent_values = []
            else:
                # Concaténer les valeurs des parents
                parent_values = [generated_values[p] for p in parents]

            # Préparer l'entrée du mécanisme
            # Assurer que le bruit est un tenseur 2D [batch, dim]
            current_noise = noise[var]
            if current_noise.dim() == 1:
                current_noise = current_noise.unsqueeze(0)

            # Concaténer les parents (si existants) et le bruit
            if parent_values:
                # Assurer que tous les tenseurs parents sont 2D
                parent_values_2d = [p.view(current_noise.shape[0], -1) for p in parent_values]
                mechanism_input = torch.cat(parent_values_2d + [current_noise], dim=1)
            else:
                mechanism_input = current_noise

            generated_values[var] = self.mechanisms[var](mechanism_input)
            
        return generated_values

    def do(self, intervention: Dict[str, torch.Tensor]) -> 'CausalGraph':
        """
        Effectue une intervention sur le graphe (opérateur 'do').
        Crée un nouveau graphe où les liens entrants vers les variables d'intervention
        sont coupés.

        Args:
            intervention: Dictionnaire des variables sur lesquelles intervenir et leurs valeurs.

        Returns:
            Un nouveau CausalGraph modifié par l'intervention.
        """
        new_graph = self.graph.copy()
        intervened_vars = list(intervention.keys())
        
        for var in intervened_vars:
            in_edges = list(new_graph.in_edges(var))
            new_graph.remove_edges_from(in_edges)
            
        # Dans une implémentation complète, les mécanismes structurels seraient aussi modifiés.
        # Ici, nous retournons simplement la structure de graphe modifiée.
        
        # Créer une nouvelle instance de CausalGraph avec le graphe modifié
        new_edges = list(new_graph.edges())
        # Note: ceci ne modifie que la structure, pas les mécanismes appris.
        # Une intervention modifie la *génération* de données, pas le modèle lui-même.
        # Le forward devrait être modifié pour gérer les interventions.
        return CausalGraph(self.variables, new_edges, self.var_dims)

    def is_d_separated(self, x: List[str], y: List[str], z: List[str]) -> bool:
        """
        Vérifie si les ensembles de variables X et Y sont d-séparés par Z.
        """
        return nx.d_separated(self.graph, set(x), set(y), set(z))

class CausalInferenceEngine:
    """
    Moteur d'inférence causale qui utilise un CausalGraph pour répondre à des requêtes.
    """
    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def _find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Trouve un ensemble d'ajustement valide en utilisant le critère de la backdoor.
        Un ensemble Z est valide si :
        1. Il bloque tous les chemins "backdoor" de T à O.
        2. Aucun noeud dans Z n'est un descendant de T.
        (Simplification: on retourne les parents du traitement)
        """
        # Une heuristique simple mais souvent efficace : ajuster pour les parents directs.
        adjustment_set = list(self.graph.graph.predecessors(treatment))
        
        # Vérification (simplifiée) : s'assurer que les noeuds ne sont pas des descendants
        descendants_of_treatment = nx.descendants(self.graph.graph, treatment)
        for var in adjustment_set:
            if var in descendants_of_treatment:
                # Cette heuristique ne fonctionne pas, il faut un algo plus complexe.
                # Pour l'instant, on lève une erreur.
                raise NotImplementedError(f"Le parent '{var}' est un descendant du traitement '{treatment}'. "
                                          "Un algorithme plus complexe de sélection d'ajustement est nécessaire.")
        
        return adjustment_set

    def estimate_causal_effect(self, 
                               treatment: str, 
                               outcome: str, 
                               samples: Dict[str, torch.Tensor],
                               treatment_value: float = 1.0
                              ) -> float:
        """
        Estime l'effet causal E[outcome | do(treatment=treatment_value)] en utilisant la formule d'ajustement par stratification.
        E[Y|do(X=x)] = sum_z [ E[Y|X=x, Z=z] * P(Z=z) ]

        Args:
            treatment: Nom de la variable de traitement.
            outcome: Nom de la variable de résultat.
            samples: Dictionnaire de données observationnelles {var_name: tensor}.
            treatment_value: La valeur spécifique de l'intervention sur le traitement.

        Returns:
            L'effet causal estimé (valeur moyenne de 'outcome').
        """
        adjustment_set = self._find_backdoor_adjustment_set(treatment, outcome)
        
        if not adjustment_set:
            # Cas simple : pas de confusion, on peut estimer directement.
            mask = samples[treatment] == treatment_value
            return samples[outcome][mask].mean().item()

        # --- Stratification ---
        # Cette implémentation suppose des variables d'ajustement discrètes.
        total_effect = 0.0
        
        # Obtenir les données pour les variables d'ajustement
        adjustment_data = torch.stack([samples[var] for var in adjustment_set], dim=1)
        
        # Trouver les strates uniques (combinaisons de valeurs pour Z)
        strata, p_strata = torch.unique(adjustment_data, return_counts=True, dim=0)
        p_strata = p_strata.float() / len(adjustment_data) # Calculer P(Z=z)

        for i, z_values in enumerate(strata):
            # Masque pour la strate courante P(Z=z)
            stratum_mask = (adjustment_data == z_values).all(dim=1)
            
            # Masque pour le traitement X=x dans la strate
            treatment_mask = (samples[treatment] == treatment_value)
            
            # Masque combiné pour E[Y | X=x, Z=z]
            combined_mask = stratum_mask & treatment_mask
            
            if combined_mask.sum() > 0:
                # Calculer l'espérance conditionnelle E[Y | X=x, Z=z]
                conditional_expectation = samples[outcome][combined_mask].mean()
            else:
                # Si aucune donnée pour cette strate, on ne peut rien dire. On l'ignore.
                # Une meilleure approche serait d'utiliser un modèle (ex: régression)
                # pour prédire la valeur manquante.
                conditional_expectation = 0 

            # Appliquer la formule : sum_z [ E[Y|X=x, Z=z] * P(Z=z) ]
            total_effect += conditional_expectation * p_strata[i]
            
        return total_effect.item()

    def counterfactual(self, 
                       evidence: Dict[str, torch.Tensor], 
                       intervention: Dict[str, torch.Tensor],
                       num_abduction_steps: int = 100,
                       lr: float = 0.01
                      ) -> Dict[str, torch.Tensor]:
        """
        Génère des contrefactuels en utilisant le processus en 3 étapes.

        Args:
            evidence: Les faits observés {variable: valeur}.
            intervention: L'intervention contrefactuelle {variable: valeur}.
            num_abduction_steps: Nombre d'itérations pour l'optimisation de l'abduction.
            lr: Taux d'apprentissage pour l'optimisation de l'abduction.

        Returns:
            Le résultat de la prédiction contrefactuelle.
        """
        # --- Étape 1: Abduction ---
        # On cherche à inférer le bruit exogène U qui correspond à l'évidence E.
        # Pour cela, on initialise le bruit aléatoirement et on l'optimise pour
        # reconstruire l'évidence.
        
        # Initialiser le bruit (requiert grad)
        noise = {
            var: torch.randn(1, self.graph.var_dims[var], requires_grad=True) 
            for var in self.graph.variables
        }
        optimizer = torch.optim.Adam(noise.values(), lr=lr)

        for _ in range(num_abduction_steps):
            optimizer.zero_grad()
            # Générer les valeurs à partir du bruit actuel
            generated_values = self.graph.forward(noise)
            # Calculer l'erreur de reconstruction par rapport à l'évidence
            loss = 0
            for var, value in evidence.items():
                loss += nn.MSELoss()(generated_values[var], value)
            
            if loss.item() < 1e-4: # Convergence
                break

            loss.backward()
            optimizer.step()
        
        # Le bruit optimisé est notre meilleure estimation de U | E
        inferred_noise = {k: v.detach() for k, v in noise.items()}

        # --- Étape 2 & 3: Action et Prédiction ---
        # On utilise le graphe, on applique l'intervention, et on propage
        # avec le bruit inféré.
        with torch.no_grad():
            counterfactual_result = self.graph.forward(
                noise=inferred_noise,
                intervention=intervention
            )

        return counterfactual_result
