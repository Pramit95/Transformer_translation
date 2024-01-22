import torch
import torch.nn as nn
import math

class inputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class positionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_length:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        #create a matrix of shape (seq_length,dmodel)
        pe = torch.zeros(seq_length,d_model)
        #create a vector of shape (seq_length,1)
        position = torch.arange(0,seq_length,dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #apply sin to even pos
        pe[:,0::2] = torch.sin(position * div_term)
        #apply cos to odd pos
        pe[:,1::2] = torch.cos(position * div_term)

        #need to apply this for entire batch of sentences, so adding a dimension
        #so now this becomes of dimensions (1,seq_length,d_model)
        pe=pe.unsqueeze(0)
        #adding this to the register buffer, as this is not a learnt param, and needs to be
        #saved to file for reuse later on
        self.register_buffer('pe',pe)

    def forward(self,x):
        #need to add this pe to every sentence, and make the added term a fixed param
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        #nn.parameter makes the param learnable
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative term
        self.bias = nn.Parameter(torch.zeros(1)) # additive term
    
    def forward(self,x):
        #calculate the mean of everything after the batch dim, so dim = -1
        #mean removes the dimensions to which is is applied by default
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha*((x-mean)/(std+self.eps))+self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_FF:int,dropout:float) -> None:
        super().__init__()
        #W1 and b1 (bias is by default true for nn.Linear)
        self.linear_1 = nn.Linear(d_model,d_FF)
        self.dropout = nn.Dropout(dropout)
        #W2 and b2
        self.linear_2 = nn.Linear(d_FF,d_model)

    def forward(self,x):
        #(Batch, seq_length, d_model) -> (Batch, seq_length, d_FF) [linear1]
        #(Batch,seq_length,d_FF) -> (Batch, seq_length,d_model) [linear2]
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float) -> None:
        super().__init__()
        assert d_model%h == 0,"d_model is not divisible by h!"
        self.d_model =d_model
        self.h = h
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model) #Wq 
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv
        self.w_o = nn.Linear(d_model,d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask):
        query = self.w_q(q) #(batch,seq_length,d_model) -> (batch,seq_length,d_model) 
        key = self.w_k(k) #(batch,seq_length,d_model) -> (batch,seq_length,d_model) 
        value = self.w_v(v) #(batch,seq_length,d_model) -> (batch,seq_length,d_model) 

        # (Batch,seq_length,d_model) -> (Batch, seq_length,h,d_k) -> (Batch,h,seq_length,d_k)
        #because we want each "head" to watch the full sentence(seq_length,d_k), but a smaller part of their embedding
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.self_attention(query,key,value,mask,self.dropout)

        #(Batch,h,seq_length,d_k) -> (Batch, seq_length, h,d_k) -> (Batch, seq_length,d_model)
        #contiguous is required for in place transformation in pytorch
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_model)

        #(Batch,seq_Length,d_model) -> (Batch, seq_length,d_model)
        return self.w_o(x)

    @staticmethod
    def self_attention(query, key, value, mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(Batch, h, seq_length, seq_length)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # second term is being returned for later visualization
        return (attention_scores @ value),attention_scores

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer): #sublayer = output from previous layer
        return x+self.dropout(sublayer(self.norm(x))) #in the paper norm is applied to sublayer, but most implementations do it this way

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self,x,src_mask): #src_mask is required to hide the padding in the encoder sequences
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        #each layer is an ecoder block
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#the decoder has a self attention block with queries, keys and values from the decoder
#and also a cross attention block, which has keys and values from the encoder output
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    #src language and target language are different, hence 2 different masks
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        #self attention for decoder queries, keys , values and mask
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,decoder_mask))
        #cross attention for decoder queries, encoder keys, values and mask
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x,encoder_output,encoder_output,encoder_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            #each layer is a decoder block
            x = layer(x,encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)

#last linear layer for projecting decoder output to current words in vocabulary
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        #apply log soft max (for numerical stability not softmax) to the last dim
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    #different input embeddings for different languages
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embed:inputEmbeddings, target_embed:inputEmbeddings,
                 src_pos:positionalEncoding,target_pos:positionalEncoding, projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.input_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt,tgt_mask):
        tgt = self.target_embed(tgt)
        tgt = self.target_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def buildTransformer(src_vocab_size:int, target_vocab_size:int, src_seq_len:int,trgt_seq_len:int, d_model:int = 512, N:int = 6,h:int= 8, dropout:float =0.1, d_ff:int =2048) -> Transformer:
    #create the embedding layers
    src_embed = inputEmbeddings(d_model, src_vocab_size)
    trgt_embed = inputEmbeddings(d_model,target_vocab_size)

    #create the positional encoding layers
    src_pos = positionalEncoding(d_model,src_seq_len,dropout)
    trgt_pos = positionalEncoding(d_model,trgt_seq_len,dropout) #can be re-used if seq_len is same

    #create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        Encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(Encoder_block)
    
    #create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        Decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(Decoder_block)
    
    #create the Encoder and the Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #create the projection layer (use target to converge to target language)
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    #create the transformer
    transformer = Transformer(encoder, decoder,src_embed,trgt_embed,src_pos,trgt_pos,projection_layer)

    #initialize the transformer so it does not start randomly
    #using xaviers, but other popular algorithms also available
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer