"""
Implémentation d'un tokenizer Byte-Pair Encoding (BPE) à partir de zéro,
inspiré de l'algorithme de Sennrich et al. (2016).
"""
import regex as re
from collections import defaultdict, Counter
import json
import os

class BPETokenizer:
    """
    Un tokenizer BPE qui s'entraîne sur un corpus de texte pour apprendre
    les fusions de sous-mots les plus efficaces.
    """
    def __init__(self, vocab_size: int, special_tokens: list = None):
        """
        Initialise le tokenizer BPE.

        Args:
            vocab_size (int): La taille maximale du vocabulaire à apprendre.
            special_tokens (list, optional): Une liste de tokens spéciaux
                                             (ex: '[PAD]', '[UNK]').
        """
        self.vocab_size = vocab_size
        self.merges = {}  # (tok1, tok2) -> nouveau_tok
        self.vocab = {}   # tok -> id
        self.inv_vocab = {} # id -> tok
        
        # Le pattern de découpage de GPT-4, très robuste
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.special_tokens = special_tokens if special_tokens else []
        self._initialize_vocab_with_special_tokens()

    def _initialize_vocab_with_special_tokens(self):
        """Initialise le vocabulaire avec les tokens spéciaux."""
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inv_vocab[i] = token

    def train(self, corpus: list[str]):
        """
        Entraîne le tokenizer sur un corpus de textes.

        Args:
            corpus (list[str]): Une liste de chaînes de caractères pour l'entraînement.
        """
        if not corpus:
            return

        # 1. Prétokenisation et calcul des fréquences des mots
        word_freqs = Counter(re.findall(self.pat, " ".join(corpus)))
        
        # 2. Initialisation du vocabulaire avec les caractères de base
        alphabet = set()
        for word in word_freqs.keys():
            alphabet.update(list(word.encode("utf-8")))
        
        # Assurer que les caractères de base sont ajoutés après les tokens spéciaux
        char_id_start = len(self.vocab)
        for i, char_byte in enumerate(sorted(list(alphabet))):
            char = bytes([char_byte]).decode("utf-8", errors="replace")
            if char not in self.vocab:
                self.vocab[char] = char_id_start + i
                self.inv_vocab[char_id_start + i] = char

        # 3. Prétokeniser le corpus en liste de listes de caractères
        splits = {word: list(word) for word in word_freqs.keys()}

        # 4. Apprendre les fusions
        while len(self.vocab) < self.vocab_size:
            # Calculer les fréquences des paires
            pair_freqs = self._get_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break
            
            # Trouver la paire la plus fréquente
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Fusionner la meilleure paire
            new_token = "".join(best_pair)
            splits = self._merge_pair(best_pair, new_token, splits, word_freqs)
            
            # Ajouter la fusion (avec son rang) et le nouveau token au vocabulaire
            self.merges[best_pair] = len(self.merges) # On stocke le rang, pas le token
            if new_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[new_token] = new_id
                self.inv_vocab[new_id] = new_token
    
    def _get_pair_freqs(self, splits, word_freqs):
        """Calcule la fréquence de chaque paire adjacente."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair_to_merge, new_token, splits, word_freqs):
        """Fusionne une paire dans toutes les occurrences du corpus splitté."""
        new_splits = {}
        for word, split in splits.items():
            if len(split) < 2:
                new_splits[word] = split
                continue

            i = 0
            new_split = []
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair_to_merge:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def encode(self, text: str) -> list[int]:
        """Encode un texte en une séquence d'IDs de tokens."""
        # Prétokenisation
        pre_tokens = re.findall(self.pat, text)
        
        token_ids = []
        for pre_token in pre_tokens:
            # Spliter le mot en caractères
            split = list(pre_token)
            
            # Appliquer les fusions apprises
            while len(split) > 1:
                pair_freqs = self._get_pair_freqs({pre_token: split}, {pre_token: 1})
                # Trouver la meilleure fusion possible (celle avec le plus petit rang)
                best_pair = min(pair_freqs, key=lambda p: self.merges.get(p, float("inf")))
                
                if best_pair not in self.merges:
                    break # Plus de fusions possibles
                
                # Reconstruire le token à partir de la paire
                new_token = "".join(best_pair)
                split = self._merge_pair(best_pair, new_token, {pre_token: split}, {pre_token: 1})[pre_token]

            # Convertir les tokens finaux en IDs
            for token in split:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Gérer les tokens inconnus (devrait être rare)
                    # On pourrait utiliser un token [UNK] s'il est défini
                    pass 
                    
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Décode une séquence d'IDs de tokens en texte."""
        tokens = [self.inv_vocab.get(id, "") for id in token_ids]
        return "".join(tokens)

    def save(self, folder_path: str):
        """Sauvegarde le vocabulaire et les fusions."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        vocab_path = os.path.join(folder_path, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # Les clés de merges sont des tuples, on les convertit en str
        merges_to_save = {" ".join(k): v for k, v in self.merges.items()}
        merges_path = os.path.join(folder_path, "merges.json")
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_to_save, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder_path: str):
        """Charge un tokenizer depuis un dossier."""
        vocab_path = os.path.join(folder_path, "vocab.json")
        merges_path = os.path.join(folder_path, "merges.json")

        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError("Impossible de trouver 'vocab.json' ou 'merges.json'")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_loaded = json.load(f)
            # Reconvertir les clés en tuples
            merges = {tuple(k.split(" ")): v for k, v in merges_loaded.items()}

        # On ne connaît pas la vocab_size d'origine mais on peut la déduire
        instance = cls(vocab_size=len(vocab))
        instance.vocab = vocab
        instance.inv_vocab = {v: k for k, v in vocab.items()}
        instance.merges = merges
        
        return instance

    def get_vocab_size(self):
        return len(self.vocab) 