import torch
from typing import Dict, Any, List

def preprocess_sft_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prétraite un exemple du dataset SFT au format chat.
    Combine les messages en un prompt et une réponse.
    Crée également un champ 'text' complet pour la construction du vocabulaire.
    """
    prompt = ""
    response = ""
    full_text = ""
    
    # Parcourir les messages pour séparer le prompt de la réponse de l'assistant
    assistant_response_started = False
    for message in example['messages']:
        content = message.get('content', '')
        role = message.get('role', '')
        
        # Concaténer tout pour la construction du vocabulaire
        full_text += content + "\n"

        if role == 'assistant':
            assistant_response_started = True
            response += content + "\n"
        elif not assistant_response_started:
            prompt += content + "\n"

    # Supprimer les espaces de fin
    example['prompt'] = prompt.strip()
    example['response'] = response.strip()
    example['text'] = full_text.strip() # Pour le build_vocab
    
    return example 