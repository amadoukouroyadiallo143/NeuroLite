import argparse
import os
import time
import json 
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    get_linear_schedule_with_warmup = None
    print("Warning: transformers library not found or too old. Learning rate scheduler will not be available.")

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain NeuroLite Model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name from Hugging Face datasets.")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="Specific configuration of the dataset.")
    parser.add_argument("--output_dir", type=str, default="models/neurolite-tiny-agi", help="Directory to save tokenizer and model checkpoints.")
    parser.add_argument("--model_config_name", type=str, default="tiny", choices=["tiny", "small", "base"], help="NeuroLite model configuration size.")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for model input.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. Set > 0 for parallel data loading if on Linux/macOS or if __main__ guarded on Windows.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before optimizer step.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for learning rate warmup.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def train_tokenizer(args, dataset):
    """Trains or loads a BPE tokenizer."""
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Training a new BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        
        def text_iterator(dataset_split):
            for i in range(0, len(dataset_split), 1000):
                yield dataset_split[i: i + 1000]["text"]

        tokenizer.train_from_iterator(text_iterator(dataset["train"]), trainer=trainer)
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer

def prepare_dataset(args, tokenizer, raw_datasets):
    """Tokenize and group texts for language modeling."""
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        # tokenizer.encode_batch returns a list of Encoding objects
        encodings = tokenizer.encode_batch(
            examples[text_column_name],
            add_special_tokens=True  # Important for model to learn special tokens if defined
        )
        # datasets.map expects a dictionary of lists
        return {"input_ids": [encoding.ids for encoding in encodings]}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset"
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part.
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy() # For causal LM, labels are shifted in the model or loss function
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {args.max_seq_length}",
    )
    return lm_datasets


class DataCollatorForLanguageModeling:
    """Collator for language modeling that handles padding and labels."""
    def __init__(self, tokenizer, mlm=False): # mlm not used here, but common for collators
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, examples):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        labels = [torch.tensor(e["labels"], dtype=torch.long) for e in examples]
        
        # Pad to the longest sequence in the batch
        # Since group_texts creates fixed-size chunks, all sequences in a batch *should* be max_seq_length
        # However, the last batch of a dataset split might be smaller if not dropped.
        # For simplicity here, we assume all are max_seq_length from group_texts.
        # If dynamic padding were needed: input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        #                               labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # -100 is ignore_index for loss

        input_ids_tensor = torch.stack(input_ids)
        labels_tensor = torch.stack(labels)
        
        # For causal LM, NeuroLiteModel's forward pass handles shifting labels if task_type is 'generation'
        # and labels are provided. So, labels can be the same as input_ids here.
        # Since group_texts creates fixed-size chunks with no padding, attention_mask can be None.
        # MultiheadAttention handles None key_padding_mask as no masking.
        return {"input_ids": input_ids_tensor, "labels": labels_tensor, "attention_mask": None}


def train_model(args, model, tokenizer, train_dataloader, eval_dataloader=None):
    """Main training loop."""
    print("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    num_training_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = None
    if get_linear_schedule_with_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        print(f"Scheduler enabled: Linear warmup ({num_warmup_steps} steps) and decay ({num_training_steps} total steps).")
    else:
        print("Scheduler disabled as 'transformers.get_linear_schedule_with_warmup' is not available.")

    # AMP GradScaler
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("CUDA detected, using Automatic Mixed Precision (AMP).")

    global_step = 0
    model.zero_grad() # Clear gradients before starting, important for gradient accumulation

    for epoch in range(args.num_train_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_train_epochs} ---")
        model.train() # Set model to training mode
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None

            if scaler: # AMP enabled
                with torch.cuda.amp.autocast():
                    outputs = model(multimodal_inputs={"input_ids": input_ids}, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] # Access loss by key
                if loss is None:
                    print("Warning: Loss is None with AMP. Check model's forward pass and label provision.")
                    continue
                loss = loss / args.gradient_accumulation_steps # Normalize loss for accumulation
                scaler.scale(loss).backward()
            else: # AMP disabled or CPU
                outputs = model(multimodal_inputs={"input_ids": input_ids}, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] # Access loss by key
                if loss is None:
                    print("Warning: Loss is None. Check model's forward pass and label provision.")
                    continue
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps # Un-normalize for logging

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler: # AMP
                    # Unscales gradients and calls optimizer.step() or skips if gradients are inf/NaN
                    scaler.step(optimizer)
                    # Updates the scale for next iteration
                    scaler.update()
                else: # No AMP
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1

                progress_bar.set_postfix({'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}' if scheduler else f'{args.learning_rate:.2e}'})

                if global_step % args.save_steps == 0: # Save checkpoint
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    # It's good practice to save the model state dict, optimizer state dict, and args
                    # model.save_pretrained(checkpoint_dir) # This might not work if model is not a PreTrainedModel from transformers
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                    with open(os.path.join(checkpoint_dir, "training_args.json"), 'w') as f:
                        json.dump(vars(args), f, indent=4)
                    
                    # Save tokenizer using its own save method if it's a Tokenizer object from 'tokenizers'
                    if hasattr(tokenizer, 'save') and callable(getattr(tokenizer, 'save')):
                         tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
                    else: # If it's a transformers tokenizer, it would be tokenizer.save_pretrained()
                        print("Warning: Tokenizer type not fully handled for saving with checkpoint. Ensure it's saved correctly.")
                    print(f"Checkpoint saved to {checkpoint_dir}")
        progress_bar.close()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

        # Optional: Evaluation step at the end of each epoch
        if eval_dataloader:
            evaluate_model(args, model, eval_dataloader, device, epoch, scaler) # Pass scaler for AMP in eval

    print("Training finished.")


def evaluate_model(args, model, eval_dataloader, device, epoch_num=0, scaler=None):
    print(f"--- Evaluating model at end of epoch {epoch_num+1} ---")
    model.eval() # Set model to evaluation mode
    total_eval_loss = 0
    progress_bar_eval = tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch_num+1}", leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for batch in progress_bar_eval:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None

            if scaler and device.type == 'cuda': # Use AMP for evaluation if enabled and on CUDA
                with torch.cuda.amp.autocast():
                    outputs = model(multimodal_inputs={"input_ids": input_ids}, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] # Access loss by key
            else:
                outputs = model(multimodal_inputs={"input_ids": input_ids}, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] # Access loss by key

            if loss is not None:
                total_eval_loss += loss.item()
            else:
                print("Warning: Eval loss is None.")
            progress_bar_eval.set_postfix({'eval_loss': f'{loss.item():.4f}' if loss else 'N/A'})
    progress_bar_eval.close()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    try:
        perplexity = torch.exp(torch.tensor(avg_eval_loss))
        print(f"Perplexity: {perplexity.item():.2f}")
    except OverflowError:
        print("Perplexity calculation resulted in overflow (loss too high).")
    
    model.train() # Set model back to training mode

def main():
    args = get_args()
    print(f"Arguments: {args}")

    # Enable anomaly detection for debugging inplace operations
    torch.autograd.set_detect_anomaly(True)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Potentially add numpy.random.seed(args.seed) and random.seed(args.seed) if used

    # Load dataset
    print(f"Loading dataset {args.dataset_name} ({args.dataset_config_name})...")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    print(f"Dataset loaded: {raw_datasets}")

    # Setup tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = train_tokenizer(args, raw_datasets)
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    # Prepare dataset for language modeling
    print(f"Preparing dataset for language modeling (grouping texts to max_seq_length: {args.max_seq_length})...")
    # Check if processed dataset exists to save time (optional)
    # For simplicity, we re-process each time here.
    train_dataset = prepare_dataset(args, tokenizer, raw_datasets.filter(lambda example, idx: idx < len(raw_datasets['train']), with_indices=True, desc="Filtering train for prepare_dataset"))['train']
    eval_dataset = prepare_dataset(args, tokenizer, raw_datasets.filter(lambda example, idx: idx < len(raw_datasets['validation']), with_indices=True, desc="Filtering validation for prepare_dataset"))['validation']

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 and torch.cuda.is_available() else False # pin_memory can speed up CPU to GPU transfer
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 and torch.cuda.is_available() else False
    )

    # Initialize model
    print("Initializing NeuroLite model...")
    if args.model_config_name == "tiny":
        config = NeuroLiteConfig.tiny()
    elif args.model_config_name == "small":
        config = NeuroLiteConfig.small()
    else: # base
        config = NeuroLiteConfig.base()
    
    # Override config with specific needs for pretraining if necessary
    config.vocab_size = tokenizer.get_vocab_size() # Ensure model vocab size matches tokenizer
    config.max_seq_length = args.max_seq_length
    config.input_projection_type = "tokenized_minhash" # Ensure this is set for text token inputs
    config.use_multimodal_input = False # Explicitly for text-only pretraining

    model = NeuroLiteModel(config, task_type="generation", tokenizer=tokenizer) # task_type="generation" for causal LM
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print(f"DataLoaders created. Train batches: {len(train_dataloader)}, Eval batches: {len(eval_dataloader)}")

    # Train the model
    train_model(args, model, tokenizer, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    # This guard is important for multiprocessing with num_workers > 0 on Windows
    main()
