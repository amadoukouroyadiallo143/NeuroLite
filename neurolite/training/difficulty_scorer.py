import numpy as np
from collections import Counter
from typing import List, Dict, Any

class DifficultyScorer:
    """
    Analyzes and scores the difficulty of data samples for curriculum learning.

    The difficulty is determined by a combination of metrics, such as sequence length
    and lexical complexity (rarity of tokens). The scorer first needs to be "fitted"
    on a dataset to learn the necessary statistics (e.g., token frequencies).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initializes the DifficultyScorer.

        Args:
            alpha (float): Weight for the lexical complexity score.
            beta (float): Weight for the sequence length score.
        """
        self.alpha = alpha
        self.beta = beta
        self.token_frequencies: Dict[int, int] = None
        self.max_lexical_score = 1.0
        self.max_length_score = 1.0

    def fit(self, all_token_ids: List[List[int]]):
        """
        Fits the scorer on the entire dataset to compute necessary statistics.
        This must be called before scoring individual samples.

        Args:
            all_token_ids (List[List[int]]): A list where each element is a list
                                               of token IDs for a single sample.
        """
        print("Fitting DifficultyScorer on the dataset...")
        
        # Calculate token frequencies
        all_tokens = [token for sample in all_token_ids for token in sample]
        if not all_tokens:
            print("Warning: No tokens found in the dataset to fit the scorer.")
            self.token_frequencies = {}
            return

        self.token_frequencies = Counter(all_tokens)
        
        # Pre-calculate max scores for normalization
        lexical_scores = [self._calculate_lexical_score(sample) for sample in all_token_ids]
        length_scores = [self._calculate_length_score(sample) for sample in all_token_ids]
        
        self.max_lexical_score = max(lexical_scores) if lexical_scores else 1.0
        self.max_length_score = max(length_scores) if length_scores else 1.0

        if self.max_lexical_score == 0: self.max_lexical_score = 1.0
        if self.max_length_score == 0: self.max_length_score = 1.0

        print("Fitting complete.")

    def _calculate_lexical_score(self, token_ids: List[int]) -> float:
        """Calculates lexical score based on token rarity (inverse frequency)."""
        if not token_ids or not self.token_frequencies:
            return 0.0
        
        # Score is the average negative log frequency of its tokens.
        # Rare tokens (low frequency) contribute more to the score.
        total_tokens = sum(self.token_frequencies.values())
        score = -sum(np.log(self.token_frequencies.get(token_id, 1) / total_tokens) for token_id in token_ids)
        return score / len(token_ids)

    def _calculate_length_score(self, token_ids: List[int]) -> float:
        """Calculates score based on sequence length."""
        return float(len(token_ids))

    def score(self, token_ids: List[int]) -> float:
        """
        Scores a single sample.

        Args:
            token_ids (List[int]): The list of token IDs for the sample.

        Returns:
            float: The calculated difficulty score.
        
        Raises:
            RuntimeError: If the scorer has not been fitted first.
        """
        if self.token_frequencies is None:
            raise RuntimeError("DifficultyScorer must be fitted on the dataset before scoring. Call .fit() first.")

        # Calculate raw scores
        lexical_score = self._calculate_lexical_score(token_ids)
        length_score = self._calculate_length_score(token_ids)
        
        # Normalize scores
        normalized_lexical = lexical_score / self.max_lexical_score
        normalized_length = length_score / self.max_length_score
        
        # Combine scores with weights
        final_score = (self.alpha * normalized_lexical) + (self.beta * normalized_length)
        
        return final_score 