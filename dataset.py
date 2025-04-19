import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer  

class BilingualDataset(Dataset):
    """A dataset class for loading bilingual data."""
    def __init__(self, ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int):
        """
        Args:
            ds (Dataset): The dataset containing bilingual data.
            tokenizer_src (Tokenizer): Tokenizer for the source language.
            tokenizer_tgt (Tokenizer): Tokenizer for the target language.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            seq_len (int): Maximum sequence length for padding/truncation.
        """
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Pre-compute special tokens for efficiency
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
        # Pre-compute pad token value for mask operations
        self.pad_token_value = self.pad_token.item()
        
        # Verify special tokens exist in the tokenizer
        if self.sos_token.item() == 0 and tokenizer_tgt.token_to_id("[SOS]") != 0:
            print("Warning: [SOS] token not found in target tokenizer")
        if self.eos_token.item() == 0 and tokenizer_tgt.token_to_id("[EOS]") != 0:
            print("Warning: [EOS] token not found in target tokenizer")
        if self.pad_token.item() == 0 and tokenizer_tgt.token_to_id("[PAD]") != 0:
            print("Warning: [PAD] token not found in target tokenizer")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        """
        Process and return a single data example with proper padding and masks.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing processed tensors ready for the model
        """
        try:
            # Get the source-target pair
            src_target_pair = self.ds[idx]
            src_text = src_target_pair['translation'][self.src_lang]
            tgt_text = src_target_pair['translation'][self.tgt_lang]

            # Tokenize the source and target texts
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

            # Truncate sequences if they're too long (instead of raising an error)
            if len(enc_input_tokens) > self.seq_len - 2:  # -2 for SOS and EOS
                enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
                print(f"Warning: Source sequence at index {idx} was truncated")
                
            if len(dec_input_tokens) > self.seq_len - 2:  # -2 for SOS and EOS
                dec_input_tokens = dec_input_tokens[:self.seq_len - 2]
                print(f"Warning: Target sequence at index {idx} was truncated")

            # Calculate padding lengths
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS (EOS goes to label)

            # Create encoder input with padding (SOS + tokens + EOS + padding)
            encoder_input = torch.cat([
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_value] * enc_num_padding_tokens, dtype=torch.int64)
            ])

            # Create decoder input with padding (SOS + tokens + padding)
            decoder_input = torch.cat([
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_value] * dec_num_padding_tokens, dtype=torch.int64)
            ])

            # Create label with padding (tokens + EOS + padding)
            label = torch.cat([
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_value] * dec_num_padding_tokens, dtype=torch.int64)
            ])

            # Verify lengths
            assert encoder_input.size(0) == self.seq_len, f"Encoder input size mismatch: {encoder_input.size(0)} vs {self.seq_len}"
            assert decoder_input.size(0) == self.seq_len, f"Decoder input size mismatch: {decoder_input.size(0)} vs {self.seq_len}"
            assert label.size(0) == self.seq_len, f"Label size mismatch: {label.size(0)} vs {self.seq_len}"
            
            # Create masks for attention mechanisms
            # Encoder padding mask: 1 for token positions, 0 for padding
            encoder_padding_mask = (encoder_input != self.pad_token_value).unsqueeze(0).unsqueeze(0).int()
            
            # Decoder padding mask: 1 for token positions, 0 for padding
            decoder_padding_mask = (decoder_input != self.pad_token_value).unsqueeze(0).unsqueeze(0).int()
            
            # Causal mask to prevent attending to future tokens
            causal_mask_tensor = causal_mask(decoder_input.size(0))
            
            # Final decoder mask combines padding mask and causal mask
            decoder_mask = decoder_padding_mask & causal_mask_tensor
            
            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'encoder_mask': encoder_padding_mask,
                'decoder_mask': decoder_mask,
                'label': label,
                'src_text': src_text,
                'tgt_text': tgt_text
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy batch or re-raise the exception
            raise

def causal_mask(size):
    """
    Create a mask for the decoder to prevent attending to future tokens (causal attention).
    
    Args:
        size (int): The size of the square attention mask
        
    Returns:
        torch.Tensor: A boolean tensor of shape (1, size, size) with True in the lower triangle
    """
    # Create a lower triangular matrix of ones (with zeros above the diagonal)
    # This ensures each position can only attend to itself and previous positions
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.bool))
    return mask