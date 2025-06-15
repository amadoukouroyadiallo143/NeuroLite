import torch
import numpy as np
from typing import List, Dict, Any, Callable

from neurolite.tokenization.tokenizer import NeuroLiteTokenizer
from .difficulty_scorer import DifficultyScorer

def get_full_text_from_messages(messages: List[Dict[str, str]]) -> str:
    """Helper to reconstruct a single text string from a list of message dicts."""
    return "".join([msg['content'] for msg in messages])

class CurriculumDataCollator:
    """
    Data collator that implements curriculum learning for chat-formatted datasets.

    It takes a raw dataset, scores each sample using a DifficultyScorer based on
    the full conversation text, and then serves batches by gradually increasing
    the data difficulty according to a pacing function.
    """
    def __init__(self, 
                 raw_dataset: List[Dict[str, Any]], # Changed from List[str]
                 tokenizer: NeuroLiteTokenizer, 
                 difficulty_scorer: DifficultyScorer,
                 pacing_function: Callable[[int], float],
                 max_seq_len: int):
        """
        Initializes the CurriculumDataCollator.

        Args:
            raw_dataset (List[Dict[str, Any]]): The raw dataset, where each item
                is a dictionary expected to have a 'messages' key.
            tokenizer (NeuroLiteTokenizer): The tokenizer instance.
            difficulty_scorer (DifficultyScorer): The scorer instance.
            pacing_function (Callable[[int], float]): A function that takes the
                current step/epoch and returns the percentage of the dataset to use.
            max_seq_len (int): The maximum sequence length for padding.
        """
        self.tokenizer = tokenizer
        self.scorer = difficulty_scorer
        self.pacing_function = pacing_function
        self.max_seq_len = max_seq_len
        
        self.current_step = 0
        # The dataset preparation is now the core of the adaptation
        self.sorted_data = self._prepare_dataset(raw_dataset)

    def _prepare_dataset(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tokenizes, scores, and sorts the entire dataset by difficulty."""
        print("Preparing dataset for curriculum learning (chat format)...")

        # 1. Extract full text and tokenize all samples
        full_texts = [get_full_text_from_messages(sample['messages']) for sample in raw_dataset]
        tokenized_samples = self.tokenizer.encode_text(full_texts)
        
        # Convertir les tenseurs en listes de listes d'entiers pour le scorer
        all_token_ids_list = [ids.tolist() for ids in tokenized_samples['input_ids']]

        # 2. Fit the scorer on the tokenized data
        self.scorer.fit(all_token_ids_list)

        # 3. Score each sample
        scored_samples = []
        for i, sample_tokens_list in enumerate(all_token_ids_list):
            score = self.scorer.score(sample_tokens_list)
            # We need to keep the original messages structure for label masking later
            scored_samples.append({
                'token_ids': sample_tokens_list, # Garder la liste pour la cohérence
                'messages': raw_dataset[i]['messages'],
                'difficulty_score': score
            })
            
        # 4. Sort the dataset by difficulty score (ascending)
        sorted_samples = sorted(scored_samples, key=lambda x: x['difficulty_score'])
        
        print(f"Dataset preparation complete. {len(sorted_samples)} samples sorted by difficulty.")
        return sorted_samples

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Creates a batch by selecting from the sorted data pool based on the
        pacing function, then tokenizes and creates labels.
        """
        # Determine the current pool of data based on the pacing function
        progress = self.pacing_function(self.current_step)
        num_samples_to_use = int(progress * len(self.sorted_data))
        num_samples_to_use = max(num_samples_to_use, min(16, len(self.sorted_data)))
        data_pool = self.sorted_data[:num_samples_to_use]
        
        batch_size = len(features)
        batch_indices = np.random.choice(len(data_pool), size=batch_size, replace=False)
        batch_samples = [data_pool[i] for i in batch_indices]

        # Process the selected batch
        input_ids_list = []
        labels_list = []

        for sample in batch_samples:
            # Reconstruct prompt and full text to find the masking length
            prompt = ""
            for msg in sample['messages']:
                if msg['role'] != 'assistant':
                    prompt += msg['content']
                else:
                    break # Stop when we hit the first assistant message
            
            full_text = get_full_text_from_messages(sample['messages'])

            # Tokenize individually to get correct lengths
            tokenized_prompt = self.tokenizer.encode_text([prompt])
            tokenized_full = self.tokenizer.encode_text([full_text])

            prompt_len = tokenized_prompt['attention_mask'].sum().item()
            
            ids = tokenized_full['input_ids'].squeeze(0) # Remove batch dim
            labels = ids.clone()
            labels[:prompt_len] = -100 # Mask the prompt

            input_ids_list.append(ids)
            labels_list.append(labels)

        # Pad sequences for the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # Truncate if longer than max_seq_len
        input_ids = input_ids[:, :self.max_seq_len]
        labels = labels[:, :self.max_seq_len]

        self.current_step += 1

        return {
            "multimodal_inputs": {
                "text": {
                    "input_ids": input_ids,
                    "attention_mask": (input_ids != self.tokenizer.pad_token_id).long()
                }
            },
            "labels": labels,
        } 