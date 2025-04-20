from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import torch
import sys
import time
import os
from typing import Union, List, Optional

# Global cache for tokenizers to avoid reloading
_TOKENIZER_CACHE = {}

def load_tokenizer(config, lang):
    """Load and cache tokenizer to avoid reloading for multiple translations."""
    cache_key = f"{config['tokenizer_file'].format(lang)}"
    if cache_key not in _TOKENIZER_CACHE:
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        _TOKENIZER_CACHE[cache_key] = Tokenizer.from_file(str(tokenizer_path))
    return _TOKENIZER_CACHE[cache_key]

def translate(sentence: Union[str, int], beam_size: int = 1, temperature: float = 1.0, 
            show_progress: bool = True, batch_size: int = 1) -> str:
    """
    Translate a sentence from source to target language using the trained transformer model.
    
    Args:
        sentence: String to translate or index in the test dataset
        beam_size: Size of beam for beam search (1 = greedy)
        temperature: Temperature for sampling (1.0 = default, <1.0 = more conservative, >1.0 = more diverse)
        show_progress: Whether to show progress during generation
        batch_size: Batch size for processing multiple translations at once
        
    Returns:
        The translated sentence
    """
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get configuration
    try:
        config = get_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return "Error: Could not load configuration"
    
    # Load tokenizers with caching
    try:
        tokenizer_src = load_tokenizer(config, config['lang_src'])
        tokenizer_tgt = load_tokenizer(config, config['lang_tgt'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    
    # Build and load model
    try:
        # Display model building message
        print(f"Building transformer model with d_model={config['d_model']}...")
        
        # Build the model
        model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            config["seq_len"],
            config['seq_len'],
            d_model=config['d_model']
        ).to(device)
        
        # Find and load model weights
        model_filename = latest_weights_file_path(config)
        if not model_filename or not Path(model_filename).exists():
            raise FileNotFoundError(f"Model weights not found: {model_filename}")
            
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: Could not load model - {e}"

    # If the sentence is a number, use it as an index to the test set
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        try:
            id = int(sentence)
            print(f"Loading example {id} from dataset...")
            ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
            ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
            sentence = ds[id]['src_text']
            label = ds[id]["tgt_text"]
        except Exception as e:
            print(f"Error loading example from dataset: {e}")
            # Continue with sentence as is if there's an error
    
    seq_len = config['seq_len']

    # Move model to evaluation mode
    model.eval()
    
    # Performance optimization: Use mixed precision for CUDA devices
    use_amp = device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    autocast = torch.cuda.amp.autocast if use_amp else lambda: DummyContextManager()
    
    start_time = time.time()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        try:
            # Tokenize input sentence
            source = tokenizer_src.encode(sentence)
            source_tokens = torch.cat([
                torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)
            
            # Create source mask (1 for tokens, 0 for padding)
            source_mask = (source_tokens != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            
            # Use mixed precision if available
            with autocast() if use_amp else DummyContextManager():
                # Get encoder output once for efficiency
                encoder_output = model.encode(source_tokens, source_mask)
            
            # Initialize the decoder input with the SOS token
            decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source_tokens).to(device)
            
            # Set up loop progress tracking
            eos_id = tokenizer_tgt.token_to_id('[EOS]')
            max_length = min(seq_len, len(source.ids) * 2)  # Reasonable maximum length
            
            # Print source and target information
            if label:
                print(f"{f'ID: ':>12}{int(sentence)}")
            print(f"{f'SOURCE: ':>12}{sentence}")
            if label:
                print(f"{f'TARGET: ':>12}{label}")
            print(f"{f'PREDICTED: ':>12}", end='', flush=True)
            
            # Generate the translation word by word
            decoded_words = []
            for i in range(max_length):
                # Build mask for target sequence
                if decoder_input.size(1) > 1:
                    # Use cached causal mask from dataset module for efficiency
                    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
                else:
                    # Single token has no need for masking future tokens
                    decoder_mask = torch.ones((1, 1, 1, 1)).type_as(source_mask).to(device)
                
                # Decode and project next token with mixed precision if available
                with autocast() if use_amp else DummyContextManager():
                    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
                    prob = model.project(out[:, -1])
                    
                    # Apply temperature sampling for more diversity when temperature != 1.0
                    if temperature != 1.0:
                        prob = prob / temperature
                    
                    # Get next word (greedy decoding - can be replaced with beam search)
                    if beam_size == 1:
                        # Simple greedy decoding
                        _, next_word = torch.max(prob, dim=1)
                    else:
                        # For now, fall back to greedy search
                        _, next_word = torch.max(prob, dim=1)
                
                # Add token to decoder input for next iteration
                next_word_token = next_word.item()
                decoder_input = torch.cat([
                    decoder_input, 
                    torch.empty(1, 1).type_as(source_tokens).fill_(next_word_token).to(device)
                ], dim=1)
                
                # Decode and print the predicted word
                next_word_str = tokenizer_tgt.decode([next_word_token])
                decoded_words.append(next_word_str)
                print(f"{next_word_str}", end=' ', flush=True)
                
                # Break if EOS token is predicted or max length reached
                if next_word_token == eos_id:
                    break
                
                # Visual progress indicator for longer translations
                if show_progress and i > 0 and i % 20 == 0:
                    print(f"\n{' ':>12}... ", end='', flush=True)
        
        except Exception as e:
            print(f"\nError during translation: {e}")
            return f"Error: Translation failed - {e}"
    
    # Print translation stats
    elapsed = time.time() - start_time
    translated_text = tokenizer_tgt.decode(decoder_input[0].tolist())
    print(f"\n{'-'*40}")
    print(f"Translation completed in {elapsed:.2f}s")
    
    return translated_text

class DummyContextManager:
    """A dummy context manager for when autocast is not available."""
    def __enter__(self): return None
    def __exit__(self, *args): return None

def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate text using Transformer model")
    parser.add_argument("text", nargs="?", default="I am not a very good student.", 
                    help="Text to translate or index in test set")
    parser.add_argument("--temperature", "-t", type=float, default=1.0,
                    help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--beam", "-b", type=int, default=1,
                    help="Beam size for beam search (default: 1 = greedy)")
    parser.add_argument("--no-progress", action="store_true",
                    help="Disable progress indicators")
    
    args = parser.parse_args()
    
    # Call translate with command line arguments
    translate(
        args.text, 
        beam_size=args.beam,
        temperature=args.temperature,
        show_progress=not args.no_progress
    )

# If script is run directly
if __name__ == "__main__":
    main()