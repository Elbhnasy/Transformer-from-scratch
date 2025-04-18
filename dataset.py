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

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize the source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add SOS and EOS tokens to the target input
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure we have enough padding tokens
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("The sequence length is too Short for the given text.")
        
        # Create encoder input with padding
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # Create decoder input with padding
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Create label with padding
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        encoder_padding_mask = (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int()
        
        # Create the decoder mask properly
        decoder_padding_mask = (decoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int()
        causal_mask_tensor = causal_mask(decoder_input.size(0))
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

def causal_mask(size):
    """Create a mask for the decoder to prevent attending to future tokens (causal attention)."""
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.bool))
    return mask