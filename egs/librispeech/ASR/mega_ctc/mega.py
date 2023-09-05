from typing import Optional, Tuple, List, Union, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    SequenceNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.models.wav2vec.utils import pad_to_multiple
from torch.nn.utils.rnn import pad_sequence

# new
from transformer import Supervisions, Transformer, encoder_padding_mask
from subsampling import Conv2dSubsampling

Supervisions = Dict[str, torch.Tensor]

class MegaLRAEncoder(Transformer):
    """
    Implementation for a Bi-directional FLASH based Sentence Encoder used
    in masked pre-trained language models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int = 1,
        vocab_size: int = 1,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        activation: str = 'silu',
        attention_activation: str = 'softmax',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        norm_type: str = 'layernorm',
        normalize_before: bool = False,
        normalize_embedding: bool = False,
        feature_dropout: bool = False,
        layerdrop: float = 0.0,
        truncation: int = None,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 256,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
        
        # transformer
        num_features: int=80,
        num_classes: int=500, #随便写一个
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        # num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        # dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        # normalize_before: bool = True,
        vgg_frontend: bool = False,
        use_feat_batchnorm: Union[float, bool] = 0.1,
    ) -> None:

        super(MegaLRAEncoder, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            normalize_before=normalize_before,
            vgg_frontend=vgg_frontend,
            use_feat_batchnorm=use_feat_batchnorm,
        )
        
    
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.embedding_dim,
                                                 self.vocab_size, self.padding_idx)

        assert not normalize_embedding or not normalize_before
        self.embed_norm = SequenceNorm(norm_type, embedding_dim, export=export) if normalize_embedding else None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_mega_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                z_dim=z_dim,
                n_dim=n_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                chunk_size=chunk_size,
                truncation=truncation,
                rel_pos_bias=rel_pos_bias,
                max_positions=self.max_seq_len,
                activation=activation,
                attention_activation=attention_activation,
                norm_type=norm_type,
                prenorm=normalize_before,
                feature_dropout=feature_dropout,
                export=export
            )
            for _ in range(self.num_layers)
        ])

        if normalize_before:
            self.final_norm = SequenceNorm(norm_type, embedding_dim, export=export)
        else:
            self.final_norm = None
            
            
        # #new 
        # self.num_features = num_features
        # self.encoder_embed = Conv2dSubsampling(num_features,hidden_dim)  #80, 512
        
        # # TODO(fangjun): remove dropout
        # # ctc
        
  
        # self.encoder_output_layer = nn.Sequential(
        #     nn.Dropout(p=dropout), nn.Linear(hidden_dim, num_classes)
        # )

    def build_embedding(self, embedding_type, embedding_dim, vocab_size, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens =(vocab_size, embedding_dim, padding_idx)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_mega_sentence_encoder_layer(
        self,
        embedding_dim,
        hidden_dim,
        ffn_hidden_dim,
        z_dim,
        n_dim,
        dropout,
        attention_dropout,
        hidden_dropout,
        chunk_size,
        truncation,
        rel_pos_bias,
        max_positions,
        activation,
        attention_activation,
        norm_type,
        prenorm,
        feature_dropout,
        export,
    ):
        return MegaSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            z_dim=z_dim,
            n_dim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            truncation=truncation,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            norm_type=norm_type,
            prenorm=prenorm,
            feature_dropout=feature_dropout,
            export=export
        )

    def forward(
        self, x: torch.Tensor, supervisions: Optional[Supervisions] = None
    ) ->  Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
 
        x = self.encoder_embed(x)  #输入torch.Size([11, 1733, 80]) 输出torch.Size([11, 432, 512])  降采样4倍! 也做 公平比较
               
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        mask = encoder_padding_mask(x.size(0), supervisions)
        if mask is not None:
            padding_mask = mask.to(x.device)  #torch.Size([11, 432])
        x = x.permute(1, 0, 2)  #
        
        # mega
        if padding_mask is not None:
            # B x N
            inverse_mask = 1.0 - padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None        
            
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
 
        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)
        
        encoder_memory = x  #torch.Size([432, 11, 512])
        x = self.ctc_output(x)  #这里x是  (N, T, C)  
        
        return x,encoder_memory,padding_mask
    
