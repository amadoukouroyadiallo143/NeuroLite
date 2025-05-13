"""
Module symbolique pour NeuroLite.
Implémente un système léger de raisonnement symbolique qui peut être intégré
à l'architecture neurale pour ajouter des capacités de raisonnement structuré
sans augmenter significativement le nombre de paramètres.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import numpy as np


class SymbolicRuleEngine:
    """
    Moteur de règles symboliques ultra-léger.
    
    Implémente un système de règles basé sur la logique de premier ordre
    qui peut être intégré dans un pipeline neuronal.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialise le moteur de règles.
        
        Args:
            rules_file: Chemin vers un fichier JSON contenant des règles prédéfinies
        """
        # Base de connaissances: stocke les faits (prédicats)
        self.facts = set()
        
        # Règles: stocke les règles d'inférence
        # Chaque règle est un tuple (prémisses, conclusion)
        self.rules = []
        
        # Charger les règles depuis un fichier si fourni
        if rules_file:
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str):
        """
        Charge les règles depuis un fichier JSON.
        
        Format attendu:
        {
            "facts": ["predicate(arg1, arg2)", ...],
            "rules": [
                {
                    "premises": ["predicate1(?x, ?y)", "predicate2(?y, ?z)"],
                    "conclusion": "predicate3(?x, ?z)"
                },
                ...
            ]
        }
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Charger les faits
            if "facts" in data:
                for fact in data["facts"]:
                    self.add_fact(fact)
                    
            # Charger les règles
            if "rules" in data:
                for rule in data["rules"]:
                    if "premises" in rule and "conclusion" in rule:
                        self.add_rule(rule["premises"], rule["conclusion"])
        except Exception as e:
            print(f"Erreur lors du chargement des règles: {e}")
    
    def add_fact(self, fact: str):
        """
        Ajoute un fait à la base de connaissances.
        
        Args:
            fact: Un prédicat, ex: "estUn(chien, animal)"
        """
        # Normaliser le fait (supprimer espaces superflus)
        fact = fact.strip()
        self.facts.add(fact)
    
    def add_rule(self, premises: List[str], conclusion: str):
        """
        Ajoute une règle d'inférence.
        
        Args:
            premises: Liste de prédicats formant les prémisses
            conclusion: Prédicat formant la conclusion
        """
        self.rules.append((premises, conclusion))
    
    def _match_predicates(self, pattern: str, predicate: str) -> Optional[Dict[str, str]]:
        """
        Tente de faire correspondre un motif de prédicat avec un prédicat concret.
        Retourne les substitutions de variables si correspondance trouvée.
        
        Args:
            pattern: Motif de prédicat avec variables "?x", ex: "estUn(?x, animal)"
            predicate: Prédicat concret, ex: "estUn(chien, animal)"
            
        Returns:
            Dictionnaire de substitutions si correspondance, None sinon
        """
        # Extraire le nom du prédicat et les arguments
        pattern_match = re.match(r'(\w+)\((.*)\)', pattern)
        predicate_match = re.match(r'(\w+)\((.*)\)', predicate)
        
        if not pattern_match or not predicate_match:
            return None
            
        pattern_name, pattern_args = pattern_match.groups()
        predicate_name, predicate_args = predicate_match.groups()
        
        # Vérifier que les noms correspondent
        if pattern_name != predicate_name:
            return None
            
        # Diviser les arguments
        pattern_args_list = [arg.strip() for arg in pattern_args.split(',')]
        predicate_args_list = [arg.strip() for arg in predicate_args.split(',')]
        
        # Vérifier que le nombre d'arguments correspond
        if len(pattern_args_list) != len(predicate_args_list):
            return None
            
        # Construire le dictionnaire de substitutions
        substitutions = {}
        for p_arg, c_arg in zip(pattern_args_list, predicate_args_list):
            # Si c'est une variable (commence par "?")
            if p_arg.startswith('?'):
                # Si cette variable a déjà été assignée, vérifier que c'est la même valeur
                if p_arg in substitutions and substitutions[p_arg] != c_arg:
                    return None
                substitutions[p_arg] = c_arg
            # Sinon, c'est une constante, vérifier l'égalité
            elif p_arg != c_arg:
                return None
                
        return substitutions
    
    def _apply_substitutions(self, predicate: str, substitutions: Dict[str, str]) -> str:
        """
        Applique des substitutions de variables à un prédicat.
        
        Args:
            predicate: Prédicat avec variables
            substitutions: Dictionnaire de substitutions
            
        Returns:
            Prédicat avec variables remplacées
        """
        # Extraire le nom du prédicat et les arguments
        predicate_match = re.match(r'(\w+)\((.*)\)', predicate)
        if not predicate_match:
            return predicate
            
        predicate_name, predicate_args = predicate_match.groups()
        
        # Diviser les arguments
        args_list = [arg.strip() for arg in predicate_args.split(',')]
        
        # Appliquer les substitutions
        for i, arg in enumerate(args_list):
            if arg in substitutions:
                args_list[i] = substitutions[arg]
                
        # Reconstruire le prédicat
        return f"{predicate_name}({', '.join(args_list)})"
    
    def infer(self, max_iterations: int = 10) -> Set[str]:
        """
        Exécute le moteur d'inférence pour dériver de nouveaux faits.
        
        Args:
            max_iterations: Nombre maximal d'itérations pour éviter les boucles infinies
            
        Returns:
            L'ensemble des faits dérivés
        """
        derived_facts = set(self.facts)  # Copie des faits existants
        
        # Boucle d'inférence
        for _ in range(max_iterations):
            new_facts = set()
            
            # Pour chaque règle
            for premises, conclusion in self.rules:
                # Liste des substitutions candidates pour chaque prémisse
                all_substitutions = []
                
                # Pour chaque prémisse, trouver toutes les correspondances possibles
                for premise in premises:
                    premise_substitutions = []
                    
                    # Chercher dans les faits dérivés
                    for fact in derived_facts:
                        substitution = self._match_predicates(premise, fact)
                        if substitution is not None:
                            premise_substitutions.append(substitution)
                    
                    # Si aucune correspondance pour cette prémisse, passer à la règle suivante
                    if not premise_substitutions:
                        break
                        
                    all_substitutions.append(premise_substitutions)
                
                # Si on a des correspondances pour toutes les prémisses
                if len(all_substitutions) == len(premises):
                    # Générer toutes les combinaisons possibles de substitutions
                    import itertools
                    for subst_combination in itertools.product(*all_substitutions):
                        # Fusionner les substitutions
                        merged_subst = {}
                        is_consistent = True
                        
                        for subst in subst_combination:
                            for var, val in subst.items():
                                if var in merged_subst and merged_subst[var] != val:
                                    is_consistent = False
                                    break
                                merged_subst[var] = val
                                
                            if not is_consistent:
                                break
                                
                        if is_consistent:
                            # Appliquer les substitutions à la conclusion
                            new_fact = self._apply_substitutions(conclusion, merged_subst)
                            if new_fact not in derived_facts:
                                new_facts.add(new_fact)
            
            # Si aucun nouveau fait dérivé, on arrête
            if not new_facts:
                break
                
            # Ajouter les nouveaux faits
            derived_facts.update(new_facts)
        
        return derived_facts


class NeuralSymbolicLayer(nn.Module):
    """
    Couche qui combine traitement neuronal et symbolique.
    
    Extrait des entités et relations des représentations latentes,
    les traite avec un moteur symbolique, puis réinjecte les résultats
    dans le flux neuronal.
    """
    
    def __init__(
        self,
        hidden_size: int,
        entity_extraction_threshold: float = 0.3,
        max_entities: int = 10,
        symbolic_rules_file: Optional[str] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.entity_extraction_threshold = entity_extraction_threshold
        self.max_entities = max_entities
        
        # Projections pour l'extraction d'entités et de relations
        self.entity_extractor = nn.Linear(hidden_size, hidden_size // 2)
        self.relation_extractor = nn.Linear(hidden_size, hidden_size // 2)
        
        # Projection pour réinjecter dans le flux neuronal
        self.output_projection = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Moteur symbolique
        self.rule_engine = SymbolicRuleEngine(symbolic_rules_file)
        
        # Petite table d'associations pour stocker les représentations des entités et prédicats
        self.entity_embeddings = {}
        self.predicate_embeddings = {}
        
    def _extract_entities(self, hidden_states: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Extrait des entités potentielles des états cachés.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple de (entités, embeddings)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Pour simplifier, on ne travaille qu'avec le premier élément du batch
        states = hidden_states[0]
        
        # Projeter pour l'extraction d'entités
        entity_logits = self.entity_extractor(states)  # [seq_len, hidden_size//2]
        
        # Calcul de score d'entité par position (simplifié)
        entity_scores = torch.norm(entity_logits, dim=1)  # [seq_len]
        
        # Sélectionner les positions avec les scores les plus élevés
        top_scores, top_indices = torch.topk(
            entity_scores, 
            min(self.max_entities, seq_len)
        )
        
        # Filtrer par seuil
        mask = top_scores > self.entity_extraction_threshold
        top_indices = top_indices[mask]
        
        # Créer des identifiants d'entités simples (E1, E2, etc.)
        entities = [f"E{i+1}" for i in range(len(top_indices))]
        
        # Récupérer les embeddings correspondants
        entity_embeddings = entity_logits[top_indices]  # [num_entities, hidden_size//2]
        
        return entities, entity_embeddings
    
    def _extract_relations(
        self, 
        hidden_states: torch.Tensor,
        entities: List[str],
        entity_embeddings: torch.Tensor
    ) -> List[str]:
        """
        Extrait des relations potentielles entre les entités.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            entities: Liste d'identifiants d'entités
            entity_embeddings: Embeddings des entités [num_entities, hidden_size//2]
            
        Returns:
            Liste de prédicats symboliques
        """
        if len(entities) < 2:
            return []
            
        batch_size, seq_len, _ = hidden_states.shape
        
        # Pour simplifier, on ne travaille qu'avec le premier élément du batch
        states = hidden_states[0]
        
        # Projeter pour l'extraction de relations
        relation_logits = self.relation_extractor(states)  # [seq_len, hidden_size//2]
        
        # Calculer similitude entre entités (produit scalaire normalisé)
        entity_embeds_norm = F.normalize(entity_embeddings, p=2, dim=1)
        similarities = torch.mm(entity_embeds_norm, entity_embeds_norm.t())  # [num_entities, num_entities]
        
        # Créer des prédicats pour les paires avec forte similitude
        predicates = []
        num_entities = len(entities)
        
        for i in range(num_entities):
            for j in range(i+1, num_entities):
                similarity = similarities[i, j].item()
                
                # Si similarité forte, créer un prédicat "similaire"
                if similarity > 0.7:
                    predicates.append(f"similaire({entities[i]}, {entities[j]})")
                # Si similarité faible mais positive, créer un prédicat "liéÀ"
                elif similarity > 0.3:
                    predicates.append(f"liéÀ({entities[i]}, {entities[j]})")
        
        return predicates
    
    def _process_symbolic(self, predicates: List[str]) -> List[str]:
        """
        Traite des prédicats avec le moteur symbolique.
        
        Args:
            predicates: Liste de prédicats
            
        Returns:
            Liste de prédicats dérivés
        """
        # Réinitialiser le moteur et ajouter les prédicats extraits
        self.rule_engine.facts.clear()
        for predicate in predicates:
            self.rule_engine.add_fact(predicate)
            
        # Inférer de nouveaux faits
        derived_facts = self.rule_engine.infer()
        
        # Ne retourner que les nouveaux prédicats
        return list(derived_facts - set(predicates))
    
    def _integrate_symbolic_results(
        self, 
        hidden_states: torch.Tensor,
        symbolic_results: List[str]
    ) -> torch.Tensor:
        """
        Réintègre les résultats symboliques dans le flux neuronal.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            symbolic_results: Liste de prédicats dérivés
            
        Returns:
            États cachés augmentés [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if not symbolic_results:
            return hidden_states
            
        # Créer un "embedding" simplifié des résultats symboliques
        # Idéalement, on apprendrait une représentation pour chaque type de prédicat
        symbolic_embedding = torch.zeros(
            hidden_size // 2, 
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Pour chaque résultat symbolique, mélanger son embedding
        for predicate in symbolic_results:
            # Hachage simple du prédicat pour initialisation déterministe
            hash_value = hash(predicate) % 10000
            torch.manual_seed(hash_value)
            
            # Générer un vecteur pseudo-aléatoire pour ce prédicat
            pred_embedding = torch.randn(
                hidden_size // 2,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            # Normaliser et ajouter à l'embedding symbolique
            pred_embedding = F.normalize(pred_embedding, p=2, dim=0)
            symbolic_embedding += pred_embedding
            
        # Normaliser l'embedding final
        if torch.norm(symbolic_embedding) > 0:
            symbolic_embedding = F.normalize(symbolic_embedding, p=2, dim=0)
            
        # Concaténer avec chaque état caché
        symbolic_expanded = symbolic_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        combined = torch.cat([hidden_states, symbolic_expanded], dim=2)
        
        # Projeter vers la dimension d'origine
        output = self.output_projection(combined)
        
        return output
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans la couche neurosymbolique.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            États cachés transformés [batch_size, seq_len, hidden_size]
        """
        # Connexion résiduelle
        residual = hidden_states
        
        # Normalisation
        hidden_states = self.norm(hidden_states)
        
        # Flux neural-symbolique (seulement si en mode évaluation ou avec probabilité)
        if not self.training or torch.rand(1).item() > 0.5:
            # Extraire entités et relations
            entities, entity_embeddings = self._extract_entities(hidden_states)
            
            if entities:
                # Extraire relations entre entités
                predicates = self._extract_relations(hidden_states, entities, entity_embeddings)
                
                # Traitement symbolique
                symbolic_results = self._process_symbolic(predicates)
                
                # Réintégrer les résultats
                hidden_states = self._integrate_symbolic_results(hidden_states, symbolic_results)
        
        # Dropout et connexion résiduelle
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states


class BayesianBeliefNetwork(nn.Module):
    """
    Réseau bayésien simple permettant l'intégration de connaissances probabilistes
    dans l'architecture neurale.
    
    Ce module permet de représenter des dépendances conditionnelles entre
    variables et d'effectuer des inférences probabilistes légères.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_variables: int = 10,
        max_parents: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_variables = num_variables
        self.max_parents = max_parents
        
        # Représentation des variables bayésiennes
        self.variable_embeddings = nn.Parameter(
            torch.randn(num_variables, hidden_size // 4)
        )
        
        # Structure du réseau bayésien (matrice d'adjacence)
        # parents[i, j] = 1 signifie que j est un parent de i
        self.register_buffer(
            "parents",
            torch.zeros(num_variables, num_variables)
        )
        
        # Tables de probabilités conditionnelles (CPT)
        # Pour simplifier, on utilise une représentation factorisée
        # plutôt qu'une table complète
        self.cpt_embeddings = nn.Parameter(
            torch.randn(num_variables, hidden_size // 2)
        )
        
        # Projections pour l'extraction et l'injection
        self.variable_extractor = nn.Linear(hidden_size, num_variables)
        self.output_projection = nn.Linear(hidden_size + num_variables, hidden_size)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialiser la structure du réseau avec quelques liens aléatoires
        self._initialize_network_structure()
        
    def _initialize_network_structure(self):
        """Initialise la structure du réseau bayésien avec des liens aléatoires."""
        for i in range(self.num_variables):
            # Pour chaque variable, choisir aléatoirement un nombre de parents
            num_parents = torch.randint(0, min(i, self.max_parents) + 1, (1,)).item()
            
            if num_parents > 0 and i > 0:
                # Choisir num_parents parmi les variables précédentes (pour éviter les cycles)
                parent_indices = torch.randperm(i)[:num_parents]
                
                # Marquer ces variables comme parents
                for parent_idx in parent_indices:
                    self.parents[i, parent_idx] = 1
    
    def _infer_probabilities(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Effectue une inférence approximative dans le réseau bayésien.
        
        Args:
            evidence: Tensor d'évidence [batch_size, num_variables]
                     Valeurs entre 0 et 1, où 1 indique évidence positive,
                     0 indique évidence négative, et valeurs intermédiaires
                     indiquent incertitude.
            
        Returns:
            Probabilités marginales mises à jour [batch_size, num_variables]
        """
        batch_size = evidence.shape[0]
        
        # Initialiser les probabilités avec l'évidence
        probs = evidence.clone()
        
        # Effectuer quelques itérations d'inférence approximative
        # (propagation de croyance simplifiée)
        for _ in range(5):  # Nombre fixe d'itérations pour simplicité
            new_probs = probs.clone()
            
            # Pour chaque variable
            for i in range(self.num_variables):
                # Si on a déjà une évidence forte, ne pas mettre à jour
                mask = (evidence[:, i] < 0.9) & (evidence[:, i] > 0.1)
                if not mask.any():
                    continue
                
                # Récupérer les parents de cette variable
                parent_indices = torch.nonzero(self.parents[i], as_tuple=True)[0]
                
                if len(parent_indices) == 0:
                    continue
                
                # Récupérer les probabilités des parents
                parent_probs = probs[:, parent_indices]
                
                # Calculer la probabilité conditionnelle approximative
                # Note: Dans une implémentation plus complète, on utiliserait
                # une vraie table CPT ou un réseau neuronal pour cette étape
                parent_embed = torch.mean(
                    self.variable_embeddings[parent_indices] * 
                    parent_probs.unsqueeze(-1).expand(-1, -1, self.hidden_size // 4),
                    dim=1
                )
                
                # Combiner avec la CPT de cette variable
                cpt_embed = self.cpt_embeddings[i]
                
                # Calcul de similitude comme approximation de la CPT
                similarity = F.cosine_similarity(
                    parent_embed,
                    cpt_embed.unsqueeze(0).expand(batch_size, -1),
                    dim=1
                )
                
                # Convertir similitude en probabilité (sigmoid)
                conditional_prob = torch.sigmoid(similarity)
                
                # Mise à jour pour les instances sans évidence forte
                new_probs[mask, i] = conditional_prob[mask]
            
            # Mettre à jour les probabilités
            probs = new_probs
        
        return probs
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le réseau bayésien.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            États cachés transformés [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Connexion résiduelle
        residual = hidden_states
        
        # Normalisation
        hidden_states = self.norm(hidden_states)
        
        # Extraire l'évidence pour les variables bayésiennes
        # (moyennée sur la dimension de séquence pour simplifier)
        evidence_logits = self.variable_extractor(hidden_states)  # [batch_size, seq_len, num_variables]
        evidence = torch.sigmoid(evidence_logits.mean(dim=1))  # [batch_size, num_variables]
        
        # Effectuer l'inférence bayésienne
        posterior_probs = self._infer_probabilities(evidence)  # [batch_size, num_variables]
        
        # Réintégrer les probabilités dans le flux neuronal
        posteriors_expanded = posterior_probs.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([hidden_states, posteriors_expanded], dim=2)
        
        # Projeter vers la dimension d'origine
        output = self.output_projection(combined)
        
        # Dropout et connexion résiduelle
        output = self.dropout(output)
        output = output + residual
        
        return output
