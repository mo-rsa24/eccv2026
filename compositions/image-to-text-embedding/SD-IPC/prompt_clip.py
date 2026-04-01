from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoder
from transformers.models.clip.configuration_clip import CLIPVisionConfig, CLIPConfig

from transformers.utils import replace_return_docstrings


class CLIPEncoderWithPrompt(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig, prompt_length: int):
        # super().__init__()
        # self.config = config
        # self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # self.gradient_checkpointing = False

        super().__init__(config)
        self.prompt_length = prompt_length

        embed_dim = config.hidden_size
        total_d_layer = config.num_hidden_layers
        self._init_prompt(prompt_length, embed_dim, total_d_layer)    

    def _init_prompt(self, num_tokens, prompt_dim, total_d_layer):
        import math
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        val = math.sqrt(6. / float(3 * num_patches + prompt_dim))  # noqa

        if total_d_layer >= 0:
            #self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            #nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = nn.LayerNorm(prompt_dim, eps=self.config.layer_norm_eps)
            self.prompt_dropout = nn.Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = nn.LayerNorm(prompt_dim, eps=self.config.layer_norm_eps)
            self.prompt_dropout = nn.Dropout(0.1)

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        B, HW, C = hidden_states.shape
        HW = HW - 1
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if idx <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[idx]).expand(B, -1, -1))
                hidden_states = torch.cat((
                    hidden_states[:, 0, :].unsqueeze(1),
                    deep_prompt_emb,
                    hidden_states[:, 1+self.prompt_length:, :]
                ), dim=1)
            else:
                hidden_states = torch.cat((
                    hidden_states[:, 0, :].unsqueeze(1),
                    hidden_states[:, -HW:, :]
                ), dim=1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        hidden_states = torch.cat((
            hidden_states[:, 0, :].unsqueeze(1),
            hidden_states[:, -HW:, :]
        ), dim=1)
        hidden_states = self.prompt_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class CLIPVisionTransformerWithPrompt(CLIPVisionTransformer):

    def __init__(self, config: CLIPVisionConfig, prompt_length):
        super().__init__(config)

        if prompt_length != 0:
            self.encoder = CLIPEncoderWithPrompt(config, prompt_length)
        else:
            self.encoder = CLIPEncoder(config)
        
class CLIPVisionModelWithPrompt(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig, prompt_length):
        super().__init__(config)

        if prompt_length != 0:
            self.vision_model = CLIPVisionTransformerWithPrompt(config, prompt_length)
        else:
            self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

#clip = CLIPVisionModelWithPrompt.from_pretrained("../clip-vit-large-patch14/", prompt_length=50)
#print(clip)
