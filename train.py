from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# For reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_all_sentences(ds, lang):
    """Extract all sentences in the specified language from the dataset."""
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """Get an existing tokenizer or build and save a new one if it doesn't exist."""
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Ensure consistent capitalization of special tokens
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Create parent directory if it doesn't exist
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    '''Load the dataset and return the train and test datasets '''
    # Performance enhancement: Use a smaller split for debugging if needed
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train[:1000]') # For debugging
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Create the train and test datasets 90% train, 10% test
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=generator)

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Calculate statistics about the dataset
    max_len_src = 0
    max_len_tgt = 0
    
    # Performance enhancement: Sample a subset for statistics to speed up processing
    sample_size = min(len(ds_raw), 1000)  # Sample at most 1000 items
    sample_indices = torch.randperm(len(ds_raw))[:sample_size]
    
    for i in sample_indices:
        item = ds_raw[i.item()]
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence (sampled): {max_len_src}')
    print(f'Max length of target sentence (sampled): {max_len_tgt}')

    # Performance enhancement: Use num_workers for faster data loading
    num_workers = min(os.cpu_count(), 4)  # Use up to 4 workers
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Speeds up host to GPU transfers
    )
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """Build and return the transformer model with the specified configuration."""
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
    )
    # Print model size for reference
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    return model

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding on the model's output.
    
    Args:
        model: The trained model
        source: Source input tensor
        source_mask: Source mask tensor
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        max_len: Maximum sequence length
        device: Device to run the model on (CPU or GPU)
        
    Returns:
        torch.Tensor: Decoded sequence of token indices
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # Pre-compute encoder output for efficiency
    encoder_output = model.encode(source, source_mask)
    
    # Initialize with start token
    decoder_input = torch.empty(1, 1, dtype=torch.long, device=device).fill_(sos_idx)
    
    # Performance enhancement: Pre-allocate tensor with maximum size for efficiency
    output_sequence = torch.full((1, max_len), tokenizer_tgt.token_to_id("[PAD]"), dtype=torch.long, device=device)
    output_sequence[0, 0] = sos_idx
    
    for i in range(1, max_len):
        # Create appropriate causal mask
        decoder_mask = causal_mask(i).type_as(source_mask).to(device)
        
        # Get model prediction
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        
        # Get next token with highest probability
        _, next_word = torch.max(prob, dim=-1)
        next_word_item = next_word.item()
        
        # Add to output sequence
        output_sequence[0, i] = next_word_item
        
        # Add to decoder input for next iteration
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
        
        # Stop if end token is generated
        if next_word_item == eos_idx:
            break
    
    # Return only the valid part of the sequence (up to the current position)
    return output_sequence[0, :i+1]

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Run validation on the model using the validation dataset.
    
    Args:
        model: The trained model to validate
        validation_ds: Validation dataset
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        max_len: Maximum sequence length
        device: Device to run the model on (CPU or GPU)
        print_msg: Function to print messages
        global_step: Global step for logging
        writer: TensorBoard writer for logging
        num_examples: Number of examples to validate
    """
    model.eval()  # Ensure model is in evaluation mode
    count = 0
    source_texts = []
    expected = []
    predicted = []
    
    # Get console width for formatting output
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().strip().split()
            console_width = int(console_width)
    except:
        console_width = 80
    
    with torch.no_grad():
        val_iterator = iter(validation_ds)
        for _ in range(num_examples):
            try:
                batch = next(val_iterator)
            except StopIteration:
                break
                
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Validation is done with batch size 1
            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"

            # Generate translation
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Get source, target, and predicted text
            source_text = batch['src_text'][0]  
            target_text = batch['tgt_text'][0]  
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Store for metrics calculation
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print comparison
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

    print_msg('-'*console_width)
            
    # Log metrics to TensorBoard if writer is available
    if writer and count > 0:
        # Calculate and log various metrics
        try:
            # Character error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation/cer', cer, global_step)
            
            # Word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation/wer', wer, global_step)
            
            # BLEU score
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation/BLEU', bleu, global_step)
            
            # Also log these metrics as text for easy reference
            writer.add_text('validation/examples', 
                        f"Step {global_step}: CER={cer:.4f}, WER={wer:.4f}, BLEU={bleu:.4f}", 
                        global_step)
            
            writer.flush()
        except Exception as e:
            print_msg(f"Error calculating metrics: {e}")
    
    # No need to return to training mode here - calling code will handle that

def compute_validation_loss(model, validation_batch, tokenizer_tgt, device, loss_fn):
    """
    Compute validation loss for a single batch.
    
    Args:
        model: The model being trained
        validation_batch: A single batch from the validation dataloader
        tokenizer_tgt: Target tokenizer
        device: Device to run the model on
        loss_fn: Loss function
        
    Returns:
        float: The loss value
    """
    # Move data to device
    encoder_input = validation_batch['encoder_input'].to(device)
    decoder_input = validation_batch['decoder_input'].to(device)
    encoder_mask = validation_batch['encoder_mask'].to(device)
    decoder_mask = validation_batch['decoder_mask'].to(device)
    label = validation_batch['label'].to(device)
    
    # Forward pass
    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj_output = model.project(decoder_output)
    
    # Calculate loss
    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
    
    return loss.item()

def train_model(config:dict):
    """
    Train the transformer model using the provided configuration.
    
    Args:
        config: Configuration dictionary with training parameters
    """
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    # Print device info
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    
    device = torch.device(device)

    # Create model folder if it doesn't exist
    model_folder = Path(f"{config['datasource']}_{config['model_folder']}")
    model_folder.mkdir(parents=True, exist_ok=True)

    # Get datasets and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Get model and move to device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Performance enhancement: Use mixed precision for faster training on compatible GPUs
    use_amp = device == 'cuda' and torch.cuda.is_device_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Setup tensorboard writer
    writer = SummaryWriter(config['experiment_name'])
    
    # Write model graph to TensorBoard
    try:
        # Create sample inputs for model visualization
        dummy_src = torch.randint(0, tokenizer_src.get_vocab_size(), (1, config['seq_len']))
        dummy_tgt = torch.randint(0, tokenizer_tgt.get_vocab_size(), (1, config['seq_len']))
        dummy_src_mask = torch.ones((1, 1, 1, config['seq_len']))
        dummy_tgt_mask = torch.ones((1, 1, config['seq_len'], config['seq_len']))
        writer.add_graph(model, (dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask))
    except Exception as e:
        print(f"Error adding model graph to TensorBoard: {e}")

    # Setup optimizer with weight decay for regularization
    # Performance enhancement: Use AdamW instead of Adam for better weight decay handling
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        eps=1e-9,
        betas=(0.9, 0.98),  # Paper values
        weight_decay=0.01  # For regularization
    )
    
    # Setup learning rate scheduler
    # Performance enhancement: Use more sophisticated scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=len(train_dataloader),
        epochs=config['num_epochs'],
        pct_start=0.1,  # Warm up for 10% of the training
        div_factor=10,  # min_lr = max_lr / div_factor
        final_div_factor=100  # final_lr = max_lr / (div_factor * final_div_factor)
    )

    # Load existing model if specified
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    
    if model_filename and Path(model_filename).exists():
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
        # Re-assign optimizer to correct device
        if device.type == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        print(f"Training from scratch")
    
    # Define the loss function with label smoothing for better generalization
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)
    
    # Define validation frequency
    validation_freq = min(len(train_dataloader) // 2, 1000)  # Validate twice per epoch or every 1000 steps
    
    # Track best validation loss for model saving
    best_val_loss = float('inf')
    
    # Main training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        # Clear GPU cache before each epoch if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        model.train()
        epoch_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        
        # Training loop
        for batch_idx, batch in enumerate(batch_iterator):
            # Move data to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                
                # Calculate loss
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimize
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Update learning rate
            scheduler.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{loss.item():6.3f}", 
                "lr": f"{scheduler.get_last_lr()[0]:6.7f}"
            })
            
            # Log to TensorBoard
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
            
            global_step += 1
            
            # Run validation at specified intervals
            if global_step % validation_freq == 0:
                model.eval()
                
                # Calculate validation loss on a subset of validation data
                val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                        if val_batch_idx >= 100:  # Limit number of validation batches for speed
                            break
                        
                        # Compute validation loss
                        batch_val_loss = compute_validation_loss(
                            model, val_batch, tokenizer_tgt, device, loss_fn
                        )
                        val_loss += batch_val_loss
                        val_count += 1
                
                # Calculate average validation loss
                avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
                
                # Log validation loss
                writer.add_scalar("validation/loss", avg_val_loss, global_step)
                
                # Run validation with example translations
                run_validation(
                    model,
                    val_dataloader,
                    tokenizer_src,
                    tokenizer_tgt,
                    config['seq_len'],
                    device,
                    lambda msg: batch_iterator.write(msg),
                    global_step,
                    writer
                )
                
                # Save best model
                if avg_val_loss < best_val_loss and val_count > 0:
                    best_val_loss = avg_val_loss
                    model_filename = get_weights_file_path(config, f"best")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'val_loss': best_val_loss
                    }, model_filename)
                    
                    batch_iterator.write(f"Best model saved with validation loss: {best_val_loss:.4f}")
                
                # Return to training mode
                model.train()
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(batch_iterator)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        
        # Save checkpoint for the epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'global_step': global_step,
            'val_loss': best_val_loss
        }, model_filename)
        
        print(f"Epoch {epoch} completed with average loss: {avg_epoch_loss:.4f}")
    
    # Final message
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Load the configuration
    config = get_config()
    
    # Train the model
    train_model(config)