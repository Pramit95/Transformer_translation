from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, toeknizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = toeknizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        #can use either src or tgt tokenizer for spl toekns
        self.sos_token = torch.tensor([toeknizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([toeknizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([toeknizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> Any:
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        #provides the input ids corresponding corresponding to each word in the original sentence
        encoder_in_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_in_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_n_paddingtokens = self.seq_len - len(encoder_in_tokens) - 2 #-2 for SOS and EOS tokens
        # -1 for decoder: during training we only add SOS token to the decoder, and eos to the labels
        decoder_n_paddingtokens = self.seq_len - len(decoder_in_tokens) - 1

        if encoder_n_paddingtokens < 0 or decoder_n_paddingtokens < 0:
            raise ValueError('Sentence is too long!')
        
        #Add special tokens to the source sentence
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_in_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_n_paddingtokens, dtype = torch.int64)

            ]
        )

        #Add special tokens to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_in_tokens,dtype =torch.int64),
                torch.tensor([self.pad_token] * decoder_n_paddingtokens, dtype = torch.int64)
            ]
        )

        #label only has eos token and padding
        label = torch.cat(
            [
                torch.tensor(decoder_in_tokens,dtype =torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_n_paddingtokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len
            #adding seq dimension+batch dimension with unsqueeze
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1,1,seq_len) & (1, seq_len,seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
#makes the decoder causal, every word can see the word before it in a sentence
def causal_mask(size):
    #torch.triu -> gets every value above the diagonal
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int)
    return mask == 0 #everything that is zero becomes true, rest false