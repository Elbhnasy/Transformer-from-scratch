import torch
import os
import torchmetrics
from dataset import causal_mask

class DummyContextManager:
    """A dummy context manager for when autocast is not available."""
    def __enter__(self): return None
    def __exit__(self, *args): return None

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

def load_tokenizer(config, lang):
    """Load and cache tokenizer to avoid reloading for multiple translations."""
    from pathlib import Path
    from tokenizers import Tokenizer
    
    # Global cache for tokenizers to avoid reloading
    if not hasattr(load_tokenizer, "_TOKENIZER_CACHE"):
        load_tokenizer._TOKENIZER_CACHE = {}
    
    cache_key = f"{config['tokenizer_file'].format(lang)}"
    if cache_key not in load_tokenizer._TOKENIZER_CACHE:
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        load_tokenizer._TOKENIZER_CACHE[cache_key] = Tokenizer.from_file(str(tokenizer_path))
    return load_tokenizer._TOKENIZER_CACHE[cache_key]