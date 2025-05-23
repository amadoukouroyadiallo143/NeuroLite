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
from typing import Dict, List, Tuple, Set, Optional, Union, Any, DefaultDict
from collections import defaultdict
import itertools # Added for itertools.product in infer


class SymbolicError(Exception):
    """Classe de base pour les erreurs dans le module symbolique."""
    pass

classMalformedFactError(SymbolicError):
    """Erreur levée lorsqu'un fait est malformé."""
    pass

class MalformedRuleError(SymbolicError):
    """Erreur levée lorsqu'une règle est malformée."""
    pass


class SymbolicRuleEngine:
    """
    Moteur de règles symboliques ultra-léger.
    
    Implémente un système de règles basé sur la logique de premier ordre
    qui peut être intégré dans un pipeline neuronal. Prend en charge la négation
    dans les prémisses et utilise un index de faits simple pour améliorer
    l'efficacité de l'inférence.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialise le moteur de règles.
        
        Args:
            rules_file: Chemin optionnel vers un fichier JSON contenant des règles 
                        et faits prédéfinis.
        
        Raises:
            FileNotFoundError: Si le fichier de règles spécifié n'est pas trouvé.
            json.JSONDecodeError: Si le fichier de règles n'est pas un JSON valide.
            SymbolicError: Pour des erreurs de format dans les règles ou faits chargés.
        """
        # Base de connaissances: stocke les faits (prédicats) indexés par nom de prédicat.
        # self.facts["predicate_name"] = {set of "predicate_name(arg1, arg2)", ...}
        self.facts: DefaultDict[str, Set[str]] = defaultdict(set)
        
        # Règles: stocke les règles d'inférence
        # Chaque règle est un tuple (prémisses, conclusion)
        self.rules: List[Tuple[List[str], str]] = []
        
        if rules_file:
            self.load_rules(rules_file)
    
    def _parse_predicate(self, predicate_str: str) -> Tuple[str, List[str], bool]:
        """
        Analyse une chaîne de prédicat pour en extraire le nom, les arguments et la négation.
        
        Args:
            predicate_str: La chaîne du prédicat (ex: "estUn(?x, animal)" ou "NOT estUn(?x, animal)").
            
        Returns:
            Tuple (nom_predicat, liste_arguments, est_negue).
            
        Raises:
            MalformedFactError: Si le prédicat est malformé.
        """
        predicate_str = predicate_str.strip()
        is_negated = False
        if predicate_str.startswith("NOT "):
            is_negated = True
            predicate_str = predicate_str[4:].strip()
            
        match = re.match(r'(\w+)\((.*)\)', predicate_str)
        if not match:
            # Gérer le cas des prédicats sans arguments (ex: "pleut")
            if re.match(r'^\w+$', predicate_str): # Nom de prédicat simple sans parenthèses
                return predicate_str, [], is_negated
            raise MalformedFactError(f"Prédicat malformé: '{predicate_str}'. "
                                     "Format attendu: 'nom(arg1, arg2, ...)' ou 'nom'.")
            
        name, args_str = match.groups()
        if not args_str: # ex: "predicat()"
            args_list = []
        else:
            args_list = [arg.strip() for arg in args_str.split(',')]
            
        return name, args_list, is_negated

    def load_rules(self, rules_file: str):
        """
        Charge les règles et les faits depuis un fichier JSON.
        
        Le format attendu du JSON est:
        {
            "facts": ["predicate(arg1, arg2)", ...],
            "rules": [
                {
                    "premises": ["predicate1(?x, ?y)", "NOT predicate2(?y, ?z)"],
                    "conclusion": "predicate3(?x, ?z)"
                },
                ...
            ]
        }
        
        Args:
            rules_file: Chemin vers le fichier JSON.
            
        Raises:
            FileNotFoundError: Si le fichier n'est pas trouvé.
            json.JSONDecodeError: Si le JSON est malformé.
            SymbolicError: Pour des erreurs de format dans les règles ou faits.
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de règles introuvable: {rules_file}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Erreur de décodage JSON dans {rules_file}: {e.msg}", e.doc, e.pos)

        if "facts" in data:
            if not isinstance(data["facts"], list):
                raise MalformedFactError("La clé 'facts' doit contenir une liste de chaînes de faits.")
            for fact_str in data["facts"]:
                if not isinstance(fact_str, str):
                    raise MalformedFactError(f"Fait non valide (doit être une chaîne): {fact_str}")
                self.add_fact(fact_str)
                    
        if "rules" in data:
            if not isinstance(data["rules"], list):
                raise MalformedRuleError("La clé 'rules' doit contenir une liste de règles.")
            for rule_dict in data["rules"]:
                if not isinstance(rule_dict, dict):
                    raise MalformedRuleError(f"Règle non valide (doit être un dictionnaire): {rule_dict}")
                if "premises" not in rule_dict or "conclusion" not in rule_dict:
                    raise MalformedRuleError("Chaque règle doit avoir 'premises' et 'conclusion'. "
                                           f"Règle problématique: {rule_dict}")
                if not isinstance(rule_dict["premises"], list) or \
                   not all(isinstance(p, str) for p in rule_dict["premises"]):
                    raise MalformedRuleError("Les 'premises' doivent être une liste de chaînes. "
                                           f"Règle problématique: {rule_dict}")
                if not isinstance(rule_dict["conclusion"], str):
                     raise MalformedRuleError("La 'conclusion' doit être une chaîne. "
                                           f"Règle problématique: {rule_dict}")
                self.add_rule(rule_dict["premises"], rule_dict["conclusion"])
    
    def add_fact(self, fact_str: str):
        """
        Ajoute un fait à la base de connaissances.
        
        Args:
            fact_str: Un prédicat représentant le fait (ex: "estUn(chien, animal)").
                      Les faits ne peuvent pas être négatifs.
        
        Raises:
            MalformedFactError: Si le fait est malformé ou négatif.
        """
        fact_str = fact_str.strip()
        if not fact_str:
            raise MalformedFactError("Un fait ne peut pas être une chaîne vide.")
        
        name, _, is_negated = self._parse_predicate(fact_str)
        if is_negated:
            raise MalformedFactError(f"Les faits ne peuvent pas être négatifs: '{fact_str}'")
            
        self.facts[name].add(fact_str)
    
    def add_rule(self, premises: List[str], conclusion_str: str):
        """
        Ajoute une règle d'inférence.
        
        Args:
            premises: Liste de prédicats (chaînes) formant les prémisses.
                      Peut inclure des prédicats négatifs (ex: "NOT estMortel(?x)").
            conclusion_str: Prédicat (chaîne) formant la conclusion. Ne peut pas être négatif.
        
        Raises:
            MalformedRuleError: Si les prémisses ou la conclusion sont malformées ou si la conclusion est négative.
        """
        if not premises:
            raise MalformedRuleError("Une règle doit avoir au moins une prémisse.")
        
        # Valider les prémisses
        for premise_str in premises:
            premise_str = premise_str.strip()
            if not premise_str:
                raise MalformedRuleError("Une prémisse ne peut pas être une chaîne vide.")
            _ = self._parse_predicate(premise_str) # Valide le format

        # Valider la conclusion
        conclusion_str = conclusion_str.strip()
        if not conclusion_str:
            raise MalformedRuleError("Une conclusion ne peut pas être une chaîne vide.")
        _, _, is_negated = self._parse_predicate(conclusion_str)
        if is_negated:
            raise MalformedRuleError(f"Les conclusions de règle ne peuvent pas être négatives: '{conclusion_str}'")
            
        self.rules.append((premises, conclusion_str))

    def _match_predicate_against_fact(
        self, 
        pattern_name: str, 
        pattern_args: List[str], 
        fact_str: str
    ) -> Optional[Dict[str, str]]:
        """
        Tente de faire correspondre un motif de prédicat (nom et arguments) avec un fait concret.
        Retourne les substitutions de variables si la correspondance est trouvée.
        
        Args:
            pattern_name: Nom du prédicat du motif.
            pattern_args: Liste des arguments du motif (peuvent être des variables "?x").
            fact_str: Fait concret (ex: "estUn(chien, animal)").
            
        Returns:
            Dictionnaire de substitutions si correspondance, None sinon.
        """
        try:
            fact_name, fact_args, fact_is_negated = self._parse_predicate(fact_str)
        except MalformedFactError: # Devrait pas arriver si add_fact valide bien
            return None 

        if fact_is_negated: # On ne matche pas contre des faits négatifs (qui ne devraient pas exister)
            return None

        if pattern_name != fact_name or len(pattern_args) != len(fact_args):
            return None
            
        substitutions = {}
        for p_arg, f_arg in zip(pattern_args, fact_args):
            if p_arg.startswith('?'):
                if p_arg in substitutions and substitutions[p_arg] != f_arg:
                    return None  # Conflit de substitution
                substitutions[p_arg] = f_arg
            elif p_arg != f_arg:
                return None  # Constante différente
                
        return substitutions

    def _apply_substitutions_to_predicate_str(self, predicate_str: str, substitutions: Dict[str, str]) -> str:
        """
        Applique des substitutions de variables à une chaîne de prédicat.
        
        Args:
            predicate_str: Chaîne du prédicat avec potentiellement des variables.
            substitutions: Dictionnaire de substitutions {variable: valeur}.
            
        Returns:
            Chaîne du prédicat avec variables remplacées.
        """
        name, args, is_negated = self._parse_predicate(predicate_str) # Peut lever MalformedFactError
        
        substituted_args = [substitutions.get(arg, arg) for arg in args]
        
        result = f"{name}({', '.join(substituted_args)})"
        if not args: # Gérer le cas comme "predicat()" ou "predicat"
            result = f"{name}" if predicate_str.endswith("()") else name

        return f"NOT {result}" if is_negated else result

    def infer(self, max_iterations: int = 10) -> Set[str]:
        """
        Exécute le moteur d'inférence pour dériver de nouveaux faits.
        
        L'ordre de traitement des prémisses dans une règle peut affecter l'efficacité.
        Une heuristique possible (non implémentée ici pour garder la complexité gérable)
        serait de traiter les prémisses les moins fréquentes ou les plus restrictives
        (moins de variables) en premier pour réduire l'espace de recherche des substitutions.
        
        Args:
            max_iterations: Nombre maximal d'itérations pour éviter les boucles infinies.
            
        Returns:
            L'ensemble de tous les faits (initiaux et dérivés).
        
        Raises:
            MalformedFactError: Si un fait dérivé est malformé (ne devrait pas arriver).
        """
        # Initialiser les faits dérivés avec les faits existants de la base de connaissances
        # On fait une copie profonde pour éviter de modifier l'original pendant l'itération.
        all_derived_facts: DefaultDict[str, Set[str]] = defaultdict(set)
        for pred_name, fact_set in self.facts.items():
            all_derived_facts[pred_name].update(fact_set)

        for iteration_count in range(max_iterations):
            newly_derived_this_iteration = set()
            
            for premises_list_str, conclusion_str in self.rules:
                # Heuristique possible : réordonner premises_list_str ici
                # pour traiter les prémisses NOT en dernier, ou les plus spécifiques en premier.
                
                # Initialiser avec une liste contenant un dictionnaire de substitution vide.
                # Chaque élément de cette liste est une solution partielle.
                consistent_substitutions_list: List[Dict[str, str]] = [{}]

                for premise_str in premises_list_str:
                    next_consistent_substitutions_list = []
                    premise_name, premise_args, is_negated = self._parse_predicate(premise_str)

                    for current_subst_set in consistent_substitutions_list:
                        # Appliquer les substitutions actuelles au motif de la prémisse
                        substituted_premise_args = [current_subst_set.get(arg, arg) for arg in premise_args]
                        
                        # Vérifier si toutes les variables dans la prémisse substituée sont liées
                        # si elles ne commencent plus par '?'
                        all_vars_bound_in_premise = not any(
                            arg.startswith('?') for arg in substituted_premise_args
                        )

                        if not is_negated:
                            # Prémisse positive : chercher des faits correspondants
                            facts_to_check = all_derived_facts.get(premise_name, set())
                            if not facts_to_check and premise_name in self.facts: # Vérifier aussi les faits initiaux si all_derived_facts est vide au début
                                facts_to_check = self.facts.get(premise_name, set())

                            for fact_candidate_str in facts_to_check:
                                # Tenter de faire correspondre la prémisse substituée (ou partiellement) avec le fait
                                new_subst = self._match_predicate_against_fact(
                                    premise_name, substituted_premise_args, fact_candidate_str
                                )
                                if new_subst is not None:
                                    # Fusionner avec les substitutions existantes
                                    merged_subst = {**current_subst_set, **new_subst}
                                    # Vérifier la cohérence (normalement gérée par _match_predicate_against_fact)
                                    # Mais une double vérification peut être utile si les variables partagent des noms
                                    is_consistent_merge = True
                                    for var, val in new_subst.items():
                                        if var in current_subst_set and current_subst_set[var] != val:
                                            # Ce cas est déjà couvert par _match_predicate_against_fact
                                            # si la variable était dans pattern_args.
                                            # Mais si pattern_args contenait une constante et new_subst
                                            # essaie de lier une variable existante différemment,
                                            # ce serait un problème.
                                            # Cependant, _match_predicate_against_fact ne devrait pas retourner
                                            # de substitution pour une variable déjà liée différemment.
                                            pass # Normalement, _match_predicate_against_fact gère ça.
                                    
                                    if is_consistent_merge:
                                        next_consistent_substitutions_list.append(merged_subst)
                        else: # Prémisse négative (NOT)
                            # La prémisse négative doit être vraie: aucun fait ne doit correspondre
                            # au motif de la prémisse négative avec les substitutions actuelles.
                            
                            # Construire le motif positif à rechercher
                            positive_pattern_args = substituted_premise_args
                            
                            match_found_for_negated_premise = False
                            facts_to_check = all_derived_facts.get(premise_name, set())
                            if not facts_to_check and premise_name in self.facts:
                                facts_to_check = self.facts.get(premise_name, set())

                            for fact_candidate_str in facts_to_check:
                                # Si un fait correspond au motif positif, alors la prémisse NOT est fausse
                                if self._match_predicate_against_fact(premise_name, positive_pattern_args, fact_candidate_str) is not None:
                                    match_found_for_negated_premise = True
                                    break
                            
                            if not match_found_for_negated_premise:
                                # Aucune correspondance trouvée, la prémisse NOT est satisfaite.
                                # Les substitutions actuelles sont toujours valides.
                                next_consistent_substitutions_list.append(current_subst_set)
                    
                    consistent_substitutions_list = next_consistent_substitutions_list
                    if not consistent_substitutions_list: # Si une prémisse échoue, la règle échoue
                        break 
                
                # Si toutes les prémisses sont satisfaites avec des substitutions cohérentes
                for final_subst in consistent_substitutions_list:
                    new_fact_str = self._apply_substitutions_to_predicate_str(conclusion_str, final_subst)
                    
                    # Vérifier que le nouveau fait n'est pas déjà présent
                    # et qu'il ne contient pas de variables non liées (ce qui ne devrait pas arriver
                    # si toutes les variables de la conclusion sont dans les prémisses ou si la règle est sûre).
                    if '?' in new_fact_str:
                        # Ceci pourrait indiquer une règle non sûre où une variable de conclusion
                        # n'est pas liée par les prémisses.
                        # print(f"Attention: Fait dérivé avec variable non liée: {new_fact_str}")
                        continue

                    conclusion_name, _, _ = self._parse_predicate(new_fact_str)
                    if new_fact_str not in all_derived_facts.get(conclusion_name, set()) and \
                       new_fact_str not in self.facts.get(conclusion_name, set()): # Vérifier aussi les faits initiaux
                        newly_derived_this_iteration.add(new_fact_str)

            if not newly_derived_this_iteration:
                break # Point fixe atteint, plus de nouveaux faits
            
            for fact_str in newly_derived_this_iteration:
                name, _, _ = self._parse_predicate(fact_str)
                all_derived_facts[name].add(fact_str)
        
        # Aplatir le dictionnaire de faits dérivés en un ensemble unique pour le retour
        final_facts_set = set()
        for fact_set in all_derived_facts.values():
            final_facts_set.update(fact_set)
            
        return final_facts_set


class NeuralSymbolicLayer(nn.Module):
    """
    Couche qui combine traitement neuronal et symbolique.
    (Le code de cette classe reste inchangé par rapport à la version précédente)
    ... (copier le reste de NeuralSymbolicLayer et BayesianBeliefNetwork ici) ...
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

        # Configuration pour les embeddings apprenables
        # Dimensions des embeddings (peuvent être rendues configurables via NeuroLiteConfig)
        self.predicate_embedding_dim = hidden_size // 4 
        self.entity_embedding_dim = hidden_size // 4
        self.symbolic_integration_dim = self.predicate_embedding_dim + 2 * self.entity_embedding_dim # Example: P_emb + E1_emb + E2_emb

        # Nombre maximum de types de prédicats et d'entités uniques à stocker
        # Ces valeurs pourraient aussi venir de la configuration.
        self.max_predicate_types = getattr(config, 'max_predicate_types', 50) 
        self.max_entities_in_vocab = getattr(config, 'max_entities_in_vocab', 200) # Max unique entity strings

        self.predicate_embedding_table = nn.Embedding(self.max_predicate_types, self.predicate_embedding_dim)
        # Pour les entités, les embeddings sont dérivés par entity_extractor. 
        # Si nous voulons des embeddings *discrets* pour les entités symboliques E1, E2, etc.,
        # alors une table d'embedding est nécessaire.
        # La tâche demande "Learnable Embeddings for Entities/Predicates".
        # Les entités extraites sont déjà des vecteurs denses.
        # Ce que nous voulons probablement, ce sont des embeddings pour les *identifiants* d'entités
        # qui apparaissent dans les prédicats symboliques.

        # Vocabulaires dynamiques pour mapper les chaînes de prédicats/entités à des indices
        self.predicate_to_idx: Dict[str, int] = {}
        self.next_predicate_idx: int = 0
        
        # Les "entités" dans les prédicats symboliques (ex: "E1", "chien") sont des chaînes.
        # Nous avons besoin d'un vocabulaire pour ces chaînes d'entités si nous voulons les intégrer via une table d'embedding.
        self.symbolic_entity_to_idx: Dict[str, int] = {}
        self.next_symbolic_entity_idx: int = 0
        self.symbolic_entity_embedding_table = nn.Embedding(self.max_entities_in_vocab, self.entity_embedding_dim)
        
        # Ajuster la projection de sortie si la dimension de l'embedding symbolique intégré change
        self.output_projection = nn.Linear(hidden_size + self.symbolic_integration_dim, hidden_size)


    def _get_or_create_idx(self, item_str: str, item_to_idx_map: Dict[str, int], next_idx_val: int, max_items: int) -> Tuple[int, int]:
        """Helper pour obtenir ou créer un indice pour un item dans un vocabulaire dynamique."""
        if item_str not in item_to_idx_map:
            if next_idx_val < max_items:
                item_to_idx_map[item_str] = next_idx_val
                next_idx_val += 1
            else:
                # Le vocabulaire est plein, utiliser un indice par défaut (ex: 0 pour 'UNK') ou gérer l'erreur
                # Pour l'instant, on réutilise l'indice 0 (pourrait être l'indice d'un token <UNK>).
                # Idéalement, initialiser l'indice 0 avec un embedding <UNK>.
                return 0, next_idx_val # Retourne l'indice <UNK> et next_idx_val inchangé
        return item_to_idx_map[item_str], next_idx_val

    def _extract_entities(self, hidden_states_batch: torch.Tensor) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """
        Extrait des entités potentielles des états cachés.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple de (entités, embeddings)
        """
        """
        Extrait des entités potentielles pour chaque item dans le batch.
        
        Args:
            hidden_states_batch: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple de (batch_entities_str_ids, batch_entity_embeddings).
            batch_entities_str_ids: Liste (par item de batch) de listes d'IDs d'entités (chaînes, ex: "E0_1").
            batch_entity_embeddings: Liste (par item de batch) de tenseurs d'embeddings d'entités extraits.
        """
        batch_size, seq_len, _ = hidden_states_batch.shape
        
        batch_entities_str_ids: List[List[str]] = []
        batch_entity_embeddings_extracted: List[List[torch.Tensor]] = [] # Embeddings originaux extraits

        for b_idx in range(batch_size):
            item_states = hidden_states_batch[b_idx]  # [seq_len, hidden_size]
            
            # Projeter pour l'extraction d'entités
            item_entity_logits = self.entity_extractor(item_states)  # [seq_len, hidden_size//2]
            
            entity_scores = torch.norm(item_entity_logits, dim=1)  # [seq_len]
            
            num_entities_to_extract = min(self.max_entities, seq_len)
            if num_entities_to_extract == 0: # cas où seq_len est 0 ou max_entities est 0
                batch_entities_str_ids.append([])
                batch_entity_embeddings_extracted.append(torch.empty(0, item_entity_logits.shape[-1], device=item_entity_logits.device)) # Tenseur vide avec la bonne dim
                continue

            top_scores, top_indices = torch.topk(
                entity_scores, 
                k=num_entities_to_extract
            )
            
            mask = top_scores > self.entity_extraction_threshold
            top_indices_filtered = top_indices[mask]
            
            current_item_entity_str_ids: List[str] = []
            current_item_entity_embeddings: List[torch.Tensor] = []

            for i, entity_idx_in_seq in enumerate(top_indices_filtered):
                # Créer un ID d'entité unique pour le moteur symbolique (ex: "B0_E1" pour batch 0, entité 1)
                # Ces IDs seront mappés à des embeddings discrets via symbolic_entity_embedding_table.
                entity_str_id = f"B{b_idx}_E{i}"
                current_item_entity_str_ids.append(entity_str_id)
                
                # Stocker l'embedding original extrait, qui pourrait être utilisé pour
                # typer l'entité ou pour des relations basées sur la similarité.
                current_item_entity_embeddings.append(item_entity_logits[entity_idx_in_seq])

            batch_entities_str_ids.append(current_item_entity_str_ids)
            if current_item_entity_embeddings:
                 batch_entity_embeddings_extracted.append(torch.stack(current_item_entity_embeddings))
            else:
                 batch_entity_embeddings_extracted.append(torch.empty(0, item_entity_logits.shape[-1], device=item_entity_logits.device))
        
        return batch_entities_str_ids, batch_entity_embeddings_extracted
    
    def _extract_relations(
        self,
        batch_entities_str_ids: List[List[str]],
        batch_entity_embeddings_extracted: List[List[torch.Tensor]]
    ) -> List[List[str]]:
        """
        Extrait des relations potentielles pour chaque item du batch.
        
        Args:
            batch_entities_str_ids: IDs de chaînes des entités extraites, par item de batch.
            batch_entity_embeddings_extracted: Embeddings originaux des entités, par item de batch.
            
        Returns:
            Liste (par item de batch) de listes de prédicats symboliques (chaînes).
        """
        batch_predicates: List[List[str]] = []

        for b_idx in range(len(batch_entities_str_ids)):
            item_entity_str_ids = batch_entities_str_ids[b_idx]
            item_entity_embeddings = batch_entity_embeddings_extracted[b_idx] # Tenseur [num_item_entities, hidden_size//2]
            
            current_item_predicates: List[str] = []
            num_item_entities = len(item_entity_str_ids)

            if num_item_entities < 2:
                batch_predicates.append(current_item_predicates)
                continue
            
            # Calculer similitude entre entités (produit scalaire normalisé)
            # S'assurer que item_entity_embeddings n'est pas vide et a plus d'un vecteur
            if item_entity_embeddings.numel() == 0 or item_entity_embeddings.shape[0] < 1:
                 batch_predicates.append(current_item_predicates)
                 continue

            entity_embeds_norm = F.normalize(item_entity_embeddings, p=2, dim=1)
            similarities = torch.mm(entity_embeds_norm, entity_embeds_norm.t())  # [num_item_entities, num_item_entities]
            
            for i in range(num_item_entities):
                for j in range(i + 1, num_item_entities): # Évite les paires dupliquées et auto-relations
                    similarity_score = similarities[i, j].item()
                    
                    # Utiliser les IDs de chaînes des entités dans les prédicats
                    entity_i_str_id = item_entity_str_ids[i]
                    entity_j_str_id = item_entity_str_ids[j]
                    
                    # Exemple de logique de création de prédicats basée sur la similarité
                    # Ces noms de prédicats ("similaire", "liéÀ") seront mappés à des embeddings apprenables.
                    if similarity_score > 0.7:
                        current_item_predicates.append(f"similaire({entity_i_str_id}, {entity_j_str_id})")
                    elif similarity_score > 0.3:
                        current_item_predicates.append(f"liéÀ({entity_i_str_id}, {entity_j_str_id})")
            
            batch_predicates.append(current_item_predicates)
            
        return batch_predicates

    def _process_symbolic(self, batch_predicates: List[List[str]]) -> List[List[str]]:
        """
        Traite des prédicats avec le moteur symbolique.
        
        Args:
            predicates: Liste de prédicats
            
        Returns:
            Liste de prédicats dérivés
        """
        """
        Traite des prédicats avec le moteur symbolique pour chaque item du batch.
        
        Args:
            batch_predicates: Liste (par item de batch) de listes de prédicats.
            
        Returns:
            Liste (par item de batch) de listes de prédicats dérivés.
        """
        batch_derived_facts: List[List[str]] = []

        # Conserver l'état original des faits persistants du moteur de règles
        original_persistent_facts = {name: set_facts.copy() for name, set_facts in self.rule_engine.facts.items()}

        for item_predicates in batch_predicates:
            # Réinitialiser les faits du moteur aux faits persistants (ceux du fichier rules.json)
            self.rule_engine.facts.clear()
            for name, set_facts in original_persistent_facts.items():
                 self.rule_engine.facts[name].update(set_facts)

            # Ajouter les faits extraits pour cet item spécifique
            transient_facts_this_item = []
            for predicate_str in item_predicates:
                try:
                    pred_name, _, is_negated = self.rule_engine._parse_predicate(predicate_str)
                    if not is_negated: # Seulement les faits positifs
                        # Éviter d'ajouter des doublons si déjà présents (par ex. depuis rules.json)
                        if predicate_str not in self.rule_engine.facts[pred_name]:
                            self.rule_engine.add_fact(predicate_str)
                            transient_facts_this_item.append(predicate_str)
                except MalformedFactError:
                    pass # Ignorer les prédicats malformés de l'extraction neurale

            # Exécuter l'inférence pour cet item
            all_facts_for_item = self.rule_engine.infer() # Contient persistants + transient + dérivés pour cet item

            # Déterminer les faits *nouvellement dérivés* pour cet item
            # Nouveaux = (tous les faits après inférence) - (faits persistants + faits transient de cet item)
            known_before_infer_this_item = set(item_predicates)
            for p_set in original_persistent_facts.values():
                known_before_infer_this_item.update(p_set)
            
            newly_derived_for_item = list(all_facts_for_item - known_before_infer_this_item)
            batch_derived_facts.append(newly_derived_for_item)

        # Restaurer les faits persistants originaux dans le moteur de règles
        self.rule_engine.facts.clear()
        for name, set_facts in original_persistent_facts.items():
            self.rule_engine.facts[name].update(set_facts)
            
        return batch_derived_facts

    def _integrate_symbolic_results(
        self, 
        hidden_states_batch: torch.Tensor, # Maintenant [batch_size, seq_len, hidden_size]
        batch_symbolic_results: List[List[str]] # Liste (par item de batch) de listes de prédicats dérivés
    ) -> torch.Tensor:
        """
        Réintègre les résultats symboliques dans le flux neuronal.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            symbolic_results: Liste de prédicats dérivés
            
        Returns:
            États cachés augmentés [batch_size, seq_len, hidden_size]
        """
        """
        Réintègre les résultats symboliques dans le flux neuronal pour chaque item du batch.
        Utilise des embeddings apprenables pour les prédicats et entités.
        
        Args:
            hidden_states_batch: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            batch_symbolic_results: Liste (par item de batch) de listes de prédicats dérivés.
            
        Returns:
            États cachés augmentés [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states_batch.shape
        device = hidden_states_batch.device
        
        # Tensor pour stocker l'embedding symbolique agrégé pour chaque item du batch
        batch_aggregated_symbolic_embeddings = torch.zeros(batch_size, self.symbolic_integration_dim, device=device, dtype=hidden_states_batch.dtype)

        for b_idx, item_symbolic_results in enumerate(batch_symbolic_results):
            if not item_symbolic_results:
                continue

            item_total_symbolic_embedding = torch.zeros(self.symbolic_integration_dim, device=device, dtype=hidden_states_batch.dtype)
            num_predicates_in_item = 0

            for predicate_str in item_symbolic_results:
                try:
                    pred_name, pred_args, is_negated = self.rule_engine._parse_predicate(predicate_str)
                    if is_negated: # On n'intègre pas les prédicats négatifs directement de cette manière
                        continue

                    # Embedding du type de prédicat
                    pred_type_idx, self.next_predicate_idx = self._get_or_create_idx(
                        pred_name, self.predicate_to_idx, self.next_predicate_idx, self.max_predicate_types
                    )
                    pred_type_embedding = self.predicate_embedding_table(torch.tensor(pred_type_idx, device=device))

                    # Embeddings des arguments (entités symboliques)
                    # Pour simplifier, on prend les deux premiers arguments. Une gestion plus complexe serait nécessaire pour arité variable.
                    arg_embeddings = []
                    for i in range(min(len(pred_args), 2)): # Max 2 args for now
                        arg_str = pred_args[i]
                        # Les arguments devraient être des IDs d'entités comme "B0_E1" ou des constantes.
                        # Si ce sont des constantes (ex: "chien"), elles sont aussi mappées à des embeddings.
                        entity_idx, self.next_symbolic_entity_idx = self._get_or_create_idx(
                            arg_str, self.symbolic_entity_to_idx, self.next_symbolic_entity_idx, self.max_entities_in_vocab
                        )
                        arg_embeddings.append(self.symbolic_entity_embedding_table(torch.tensor(entity_idx, device=device)))
                    
                    # S'assurer qu'on a toujours 2 embeddings d'arguments (même si vides/padding si moins de 2 args)
                    while len(arg_embeddings) < 2:
                        # Utiliser un embedding de padding (ex: un vecteur de zéros ou un embedding <PAD> dédié)
                        arg_embeddings.append(torch.zeros(self.entity_embedding_dim, device=device, dtype=hidden_states_batch.dtype))

                    # Concaténer les embeddings (pred_type_emb, arg1_emb, arg2_emb)
                    # La dimension doit correspondre à self.symbolic_integration_dim
                    current_predicate_full_embedding = torch.cat([pred_type_embedding] + arg_embeddings, dim=0)
                    
                    item_total_symbolic_embedding += current_predicate_full_embedding
                    num_predicates_in_item += 1
                
                except MalformedFactError: # Ignorer les prédicats malformés
                    continue
            
            if num_predicates_in_item > 0:
                # Moyenne des embeddings des prédicats pour cet item
                batch_aggregated_symbolic_embeddings[b_idx] = item_total_symbolic_embedding / num_predicates_in_item
        
        # Normaliser l'embedding symbolique agrégé pour chaque item (optionnel mais souvent utile)
        # batch_aggregated_symbolic_embeddings = F.normalize(batch_aggregated_symbolic_embeddings, p=2, dim=1)

        # Étendre et concaténer avec les états cachés originaux
        symbolic_expanded = batch_aggregated_symbolic_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([hidden_states_batch, symbolic_expanded], dim=2)
        
        # Projeter vers la dimension d'origine
        output = self.output_projection(combined)
        
        return output
        
    def forward(self, hidden_states_batch: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans la couche neurosymbolique.
        
        Args:
            hidden_states: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            États cachés transformés [batch_size, seq_len, hidden_size]
        """
        """
        Passage avant dans la couche neurosymbolique, gérant les batches.
        
        Args:
            hidden_states_batch: Tensor d'états cachés [batch_size, seq_len, hidden_size]
            
        Returns:
            États cachés transformés [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states_batch
        normalized_hidden_states = self.norm(hidden_states_batch)
        
        final_hidden_states = normalized_hidden_states.clone() # Pour les items où le symbolique n'est pas appliqué

        # Condition pour appliquer le module symbolique (ex: en eval, ou stochastiquement)
        apply_symbolic = not self.training or torch.rand(1).item() > 0.5 # TODO: Configurable

        if apply_symbolic:
            # Extraire entités et relations pour tout le batch
            batch_entities_str_ids, batch_entity_embeddings_extracted = self._extract_entities(normalized_hidden_states)
            batch_initial_predicates = self._extract_relations(batch_entities_str_ids, batch_entity_embeddings_extracted)
            
            # Traitement symbolique pour chaque item du batch
            batch_derived_predicates = self._process_symbolic(batch_initial_predicates)
            
            # Réintégrer les résultats symboliques (seulement si des prédicats dérivés existent pour au moins un item)
            # _integrate_symbolic_results gère déjà les items sans résultats.
            if any(batch_derived_predicates):
                 final_hidden_states = self._integrate_symbolic_results(normalized_hidden_states, batch_derived_predicates)
            # Si aucun prédicat dérivé pour aucun item, final_hidden_states reste normalized_hidden_states.
            # La projection self.output_projection est appliquée dans _integrate_symbolic_results.
            # Si _integrate_symbolic_results n'est pas appelé car pas de résultats,
            # il faut s'assurer que la dimension est correcte ou que la projection est optionnelle.
            # Dans la version actuelle, si _integrate_symbolic_results n'est pas appelé,
            # la dimension de `final_hidden_states` reste `hidden_size`.
            # Si `_integrate_symbolic_results` est appelé, il retourne `hidden_size` après projection.
            # Cela semble correct.

        # Dropout et connexion résiduelle
        output_states = self.dropout(final_hidden_states)
        output_states = output_states + residual # Ajout de la connexion résiduelle
        
        return output_states


class BayesianBeliefNetwork(nn.Module):
    """
    Réseau bayésien simple permettant l'intégration de connaissances probabilistes
    dans l'architecture neurale.
    (Le code de cette classe reste inchangé par rapport à la version précédente)
    ...
    """
    
    def __init__(
        self,
        config: Any, # Using Any to avoid direct NeuroLiteConfig dependency here if run standalone
        hidden_size: int, # Kept for direct use if config not fully specified
        # num_variables: int = 10, # Now from config
        # max_parents: int = 3, # Now from config
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.config = config
        self.hidden_size = hidden_size
        self.num_variables = getattr(config, 'num_bayesian_variables', 10)
        self.max_parents = getattr(config, 'max_parents_bayesian', 3) # Added a default config name
        self.dropout_rate = dropout_rate

        # Dimension des embeddings pour les variables et CPTs (doivent être compatibles pour la similarité)
        self.var_embedding_dim = hidden_size // 4
        self.cpt_embedding_dim = hidden_size // 4 # Assumed to be same as var_embedding_dim

        self.variable_embeddings = nn.Parameter(
            torch.randn(self.num_variables, self.var_embedding_dim)
        )
        
        # Structure du réseau bayésien (matrice d'adjacence)
        # self.parents[i, j] = 1 signifie que j est un parent de i
        # self.children[i, j] = 1 signifie que j est un enfant de i
        self.register_buffer("parents_matrix", torch.zeros(self.num_variables, self.num_variables, dtype=torch.bool))
        self.register_buffer("children_matrix", torch.zeros(self.num_variables, self.num_variables, dtype=torch.bool))

        self.cpt_embeddings = nn.Parameter(
            torch.randn(self.num_variables, self.cpt_embedding_dim)
        )
        
        self.variable_extractor = nn.Linear(hidden_size, self.num_variables)
        self.output_projection = nn.Linear(hidden_size + self.num_variables, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self._initialize_network_structure()
        self.topological_order = self._get_topological_order()
        
    def _initialize_network_structure(self):
        """
        Initialise la structure du réseau bayésien.
        Utilise une structure prédéfinie depuis la config si fournie,
        sinon génère une structure aléatoire (assurant un DAG).
        """
        predefined_structure = getattr(self.config, 'bayesian_network_structure', None)
        
        if predefined_structure and isinstance(predefined_structure, list):
            for parent_idx, child_idx in predefined_structure:
                if 0 <= parent_idx < self.num_variables and 0 <= child_idx < self.num_variables:
                    # Ensure no self-loops from config
                    if parent_idx == child_idx: continue 
                    self.parents_matrix[child_idx, parent_idx] = True
                    self.children_matrix[parent_idx, child_idx] = True
                else:
                    print(f"Warning: Invalid parent/child index in predefined Bayesian structure: ({parent_idx}, {child_idx})")
            # Basic cycle check for predefined structure (not exhaustive, but catches direct parent > child issues if sorted)
            # A full cycle check would be more complex (e.g., DFS).
            # For now, we assume predefined structure aims to be a DAG.
        else:
            # Random DAG generation: iterate through variables, for each, pick random parents from *preceding* variables
            for i in range(self.num_variables):
                num_potential_parents = min(i, self.max_parents) # Parents must have smaller index
                if num_potential_parents == 0:
                    continue
                
                actual_num_parents = torch.randint(0, num_potential_parents + 1, (1,)).item()
                if actual_num_parents > 0:
                    parent_indices = torch.randperm(i)[:actual_num_parents]
                    for p_idx in parent_indices:
                        self.parents_matrix[i, p_idx] = True
                        self.children_matrix[p_idx, i] = True
    
    def _get_topological_order(self) -> List[int]:
        """Calcule un ordre topologique des variables. Utilise Kahn's algorithm."""
        order = []
        in_degree = self.parents_matrix.sum(dim=1).tolist()
        queue = [i for i, deg in enumerate(in_degree) if deg == 0]
        
        visited_count = 0
        while queue:
            u = queue.pop(0)
            order.append(u)
            visited_count += 1
            
            for v in torch.nonzero(self.children_matrix[u], as_tuple=True)[0].tolist():
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if visited_count != self.num_variables:
            # This indicates a cycle, which shouldn't happen with random init or valid predefined DAG
            # Fallback to simple numerical order if cycle detected, though inference might be flawed.
            print("Warning: Cycle detected in Bayesian network structure or not all nodes reachable. Falling back to numerical order.")
            return list(range(self.num_variables))
        return order

    def _get_conditional_prob(self, var_idx: int, parent_states: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Approximation de P(var_idx=1 | parent_states) en utilisant les embeddings.
        parent_states: Tensor [batch_size, num_parents_for_var_idx] contenant les états (0 ou 1) des parents.
        """
        parent_indices = torch.nonzero(self.parents_matrix[var_idx], as_tuple=True)[0]

        if len(parent_indices) == 0: # Pas de parents
            # Retourner une probabilité a priori (ex: 0.5, ou apprise)
            # Pour l'instant, on utilise l'embedding de la CPT de la variable elle-même,
            # comparé à un vecteur de zéros ou un embedding "sans parent".
            # Cela est très heuristique. Un prior appris serait mieux.
            # Let's use a sigmoid over a learnable bias for root nodes.
            # For now, using cpt_embedding dot product with itself (normalized) as a proxy for prior.
            # This part needs more principled handling of priors.
            # A simple prior:
            # return torch.sigmoid(self.cpt_embeddings[var_idx].mean()).expand(batch_size)
            # For now, let's keep it similar to when parents are present but use a zero-vector for parent_embed_mean
            parent_embed_mean = torch.zeros(batch_size, self.var_embedding_dim, device=self.cpt_embeddings.device)
        else:
            # parent_states: [batch_size, num_parents]
            # self.variable_embeddings[parent_indices]: [num_parents, var_embedding_dim]
            
            # We need to "activate" parent embeddings based on their state (0 or 1).
            # A simple way: multiply embeddings by states. If state is 0, embedding contribution is 0.
            active_parent_embeddings = self.variable_embeddings[parent_indices].unsqueeze(0) * \
                                       parent_states.unsqueeze(-1).float() # [batch_size, num_parents, var_embedding_dim]
            
            parent_embed_mean = torch.mean(active_parent_embeddings, dim=1) # [batch_size, var_embedding_dim]

        # Comparer avec l'embedding CPT de la variable actuelle
        cpt_embed_for_var = self.cpt_embeddings[var_idx] # [cpt_embedding_dim]
        
        # Assumed var_embedding_dim == cpt_embedding_dim
        similarity = F.cosine_similarity(
            parent_embed_mean,
            cpt_embed_for_var.unsqueeze(0).expand(batch_size, -1),
            dim=1
        )
        return torch.sigmoid(similarity * 5.0) # Scaling factor for sigmoid sharpness


    def _infer_probabilities(self, evidence: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
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
        """
        Effectue une inférence approximative par Likelihood Weighting.
        
        Args:
            evidence: Tensor d'évidence [batch_size, num_variables].
                      Les valeurs sont des probabilités (0 à 1).
                      Une valeur > 0.5 est considérée comme évidence positive (état 1),
                      < 0.5 comme évidence négative (état 0).
                      ~0.5 signifie pas d'évidence claire pour cette variable.
            num_samples: Nombre d'échantillons à générer pour l'estimation.
            
        Returns:
            Probabilités marginales mises à jour [batch_size, num_variables]
        """
        batch_size = evidence.shape[0]
        device = evidence.device

        # Initialiser les accumulateurs pour les probabilités postérieures
        # weighted_sum_true[b,v] = sum of weights where var v is true in sample for batch b
        weighted_sum_true = torch.zeros(batch_size, self.num_variables, device=device)
        total_weight = torch.zeros(batch_size, device=device)

        # Déterminer quelles variables sont des variables d'évidence pour chaque item du batch
        # Evidence is where evidence value is not ~0.5.
        is_evidence_var = (evidence < 0.4) | (evidence > 0.6)
        evidence_values_binary = (evidence > 0.5).long() # Convertir l'évidence en états binaires 0 ou 1

        for _ in range(num_samples):
            sample_states = torch.zeros(batch_size, self.num_variables, device=device, dtype=torch.long)
            sample_weights = torch.ones(batch_size, device=device)

            for var_idx in self.topological_order:
                parent_indices = torch.nonzero(self.parents_matrix[var_idx], as_tuple=True)[0]
                
                current_parent_states = sample_states[:, parent_indices] if len(parent_indices) > 0 else \
                                        torch.empty(batch_size, 0, device=device, dtype=torch.long)

                # Probabilité P(var_idx=1 | parents)
                prob_true = self._get_conditional_prob(var_idx, current_parent_states, batch_size) # [batch_size]

                if is_evidence_var[:, var_idx].any(): # Si cette variable est une évidence pour au moins un item du batch
                    # Obtenir la valeur d'évidence pour cette variable
                    fixed_state_for_evidence = evidence_values_binary[:, var_idx] # [batch_size]
                    
                    # Calculer P(var=evidence_val | parents)
                    prob_of_evidence_state = torch.where(fixed_state_for_evidence == 1, prob_true, 1.0 - prob_true)
                    
                    # Mettre à jour le poids de l'échantillon pour les items où var_idx est une évidence
                    # sample_weights[is_evidence_var[:, var_idx]] *= prob_of_evidence_state[is_evidence_var[:, var_idx]]
                    # More robust:
                    active_mask = is_evidence_var[:, var_idx]
                    sample_weights = torch.where(active_mask, sample_weights * prob_of_evidence_state, sample_weights)


                    # Fixer l'état de la variable à la valeur d'évidence
                    # sample_states[:, var_idx] = fixed_state_for_evidence
                    sample_states[:, var_idx] = torch.where(active_mask, fixed_state_for_evidence, sample_states[:, var_idx]) # Only fix if evidence

                    # Pour les items où ce n'est PAS une évidence, échantillonner normalement
                    if not active_mask.all(): # If there are items where var_idx is NOT evidence
                        not_evidence_mask = ~active_mask
                        sampled_state_for_non_evidence = torch.bernoulli(prob_true[not_evidence_mask]).long()
                        sample_states[not_evidence_mask, var_idx] = sampled_state_for_non_evidence
                else: # Pas une variable d'évidence pour aucun item du batch
                    # Échantillonner l'état de la variable
                    sampled_state = torch.bernoulli(prob_true).long() # [batch_size]
                    sample_states[:, var_idx] = sampled_state
            
            # Ajouter les états pondérés aux accumulateurs
            # Pour chaque variable v, si sample_states[b,v] == 1, ajouter sample_weights[b] à weighted_sum_true[b,v]
            weighted_sum_true += sample_states.float() * sample_weights.unsqueeze(1)
            total_weight += sample_weights

        # Calculer les probabilités postérieures
        # P(V=1|E) = Sum(weights where V=1) / Sum(all weights)
        # Éviter la division par zéro si total_weight est nul pour certains items du batch
        posterior_probs = torch.zeros_like(weighted_sum_true)
        safe_total_weight = torch.where(total_weight <= 1e-6, torch.ones_like(total_weight), total_weight) # Avoid div by zero

        posterior_probs = weighted_sum_true / safe_total_weight.unsqueeze(1)
        
        # Pour les variables d'évidence, la probabilité est l'évidence elle-même (ou proche de 0 ou 1)
        # On peut choisir de retourner l'évidence directement pour ces variables, ou la probabilité estimée.
        # L'estimation devrait converger vers l'évidence.
        # Pour l'instant, on retourne les probabilités estimées pour toutes.
        # Si une variable était une évidence, sa probabilité devrait être proche de 0 ou 1.
        # On peut réappliquer l'évidence pour s'assurer qu'elle est respectée.
        final_probs = torch.where(is_evidence_var, evidence_values_binary.float(), posterior_probs)

        return final_probs

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
