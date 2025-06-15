"""
Data Collator pour l'entraînement supervisé (SFT) sur des formats de chat,
adapté pour le NeuroLiteTokenizer.
"""
from dataclasses import dataclass
from typing import Dict, List, Any
import torch

from neurolite.tokenization.tokenizer import NeuroLiteTokenizer

@dataclass
class SFTDataCollator:
    """
    Data collator qui prend en charge la tokenisation et la création de labels
    pour l'entraînement supervisé, en utilisant le NeuroLiteTokenizer.

    Il gère nativement les formats de conversation (liste de messages),
    les tokenize en tokens discrets, et masque la perte pour les prompts
    de l'utilisateur.
    """
    tokenizer: NeuroLiteTokenizer
    max_length: int = 1024
    label_ignore_index: int = -100

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        input_ids_list = []
        labels_list = []

        for example in examples:
            messages = example['messages']
            
            # Reconstruire le prompt et le texte complet à partir des messages
            prompt_text = ""
            full_text = ""
            for msg in messages:
                content = msg['content']
                full_text += content
                if msg['role'] != 'assistant':
                    prompt_text += content
            
            # Tokenize pour obtenir les IDs et calculer la longueur du prompt
            # Note: nous tokenisons ici le texte brut. Le tokenizer interne le gère.
            tokenized_full = self.tokenizer.encode_text([full_text], max_length=self.max_length)
            tokenized_prompt = self.tokenizer.encode_text([prompt_text], max_length=self.max_length)

            input_ids = tokenized_full['input_ids'].squeeze(0) # enlever la dim de batch
            prompt_len = tokenized_prompt['attention_mask'].sum().item()

            # Créer les labels en masquant le prompt
            labels = input_ids.clone()
            labels[:prompt_len] = self.label_ignore_index

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Padder les séquences pour former un batch
        final_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        final_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=self.label_ignore_index
        )
        
        # Créer le masque d'attention final à partir des IDs paddés
        attention_mask = (final_input_ids != self.tokenizer.pad_token_id).long()
            
        return {
            "multimodal_inputs": {
                "text": {
                    "input_ids": final_input_ids,
                    "attention_mask": attention_mask
                }
            },
            "labels": final_labels,
        } 