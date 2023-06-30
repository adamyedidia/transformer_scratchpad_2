from example_sentences import EXAMPLE_SENTENCES, EXAMPLE_QUESTIONS, EXAMPLE_MIX, SEMICOLON_EXAMPLE_SENTENCES, \
    EXAMPLE_SUBJECT_BREAKDOWNS, EXAMPLE_SUBJECT_BREAKDOWNS_KEYS, EXAMPLE_SUBJECT_VERB_BREAKDOWNS_KEYS, \
    EXAMPLE_SUBJECT_VERB_BREAKDOWNS

IN_COLAB = False
M1_MAC = True

import random
import pickle
import einops
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from transformer_lens import loading_from_pretrained as loading
from transformer_lens.utils import gelu_new
from dataclasses import dataclass
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import math
import tiktoken
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LIST_OF_ALL_TOKENS = []
enc = tiktoken.get_encoding('r50k_base')
i = 0
for i in range(50257):
    LIST_OF_ALL_TOKENS.append(enc.decode([i]))


def alphabetic_rank(word):
    base = ord('a')
    rank = 0
    current_letter_value = 1
    for char in word.lower():
        if char.isalpha():
            rank += current_letter_value * (ord(char) - base + 1)
        else:
            raise ValueError(f"Non-alphabetic character '{char}' found in word")

        current_letter_value *= 1 / 26
    return rank


# import IPython
# IPython.embed()

def visualize_tensor(tensor, title, dim=0, cmap="viridis", max_cols=5, scaling_factor=4):
    tensor = tensor.detach().cpu().numpy()
    num_slices = tensor.shape[dim]
    num_rows = int(np.ceil(num_slices / max_cols))
    num_cols = min(num_slices, max_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(scaling_factor * max_cols, scaling_factor * num_rows),
                             constrained_layout=True)

    for i in range(num_slices):
        row, col = divmod(i, max_cols)

        if num_rows > 1:
            ax = axes[row, col]
        else:
            if num_cols == 1:
                ax = axes
            else:
                ax = axes[col]

        ax.imshow(np.take(tensor, i, axis=dim), cmap=cmap)
        ax.set_title(f"{title} (slice {i})")
        ax.axis("off")

    for i in range(num_slices, num_rows * max_cols):
        row, col = divmod(i, max_cols)
        if num_rows > 1:
            ax = axes[row, col]
        else:
            if num_cols == 1:
                ax = axes
            else:
                ax = axes[col]
        ax.axis("off")

    # plt.show()


# model_name = "gpt2-large"
model_name = "gpt2-small"
# model_name = "pythia-70m-v0"

reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                   center_writing_weights=False)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)


def cuda(x):
    return x.to('cpu') if M1_MAC else x.cuda()


tokens = cuda(tokens)
logits, cache = reference_gpt2.run_with_cache(tokens)

log_probs = logits.log_softmax(dim=-1)
probs = logits.log_softmax(dim=-1)

next_token = logits[0, -1].argmax(dim=-1)

next_tokens = torch.cat(
    [tokens, torch.tensor(next_token, device='cpu' if M1_MAC else 'cuda', dtype=torch.int64)[None, None]], dim=-1)
new_logits = reference_gpt2(next_tokens)

for activation_name, activation in cache.cache_dict.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(activation_name, activation.shape)

for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(name, param.shape)


# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
# print(reference_gpt2.cfg)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs) -> Config:
    return Config(
        **{k: v for k, v in loading.get_pretrained_model_config(model_name,
                                                                **kwargs).to_dict().items() if k in [
               'd_model',
               'layer_norm_eps',
               'd_vocab',
               'init_range',
               'n_ctx',
               'd_head',
               'd_mlp',
               'n_heads',
               'n_layers',
           ]})


cfg = get_basic_config(model_name)

# print(loading.get_pretrained_model_config(model_name))


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randn(shape))
    # print("Input shape:", random_input.shape)
    output = layer(random_input)
    # print("Output shape:", output.shape)
    # print()
    return output


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randint(100, 1000, shape))
    # print("Input shape:", random_input.shape)
    output = layer(random_input)
    # print("Output shape:", output.shape)
    # print()
    return output


def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict=cache.cache_dict):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    # Allow inputs of strings or tensors
    if isinstance(input_name, str):
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    # print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    # print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    # print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    # print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct")
    return output


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1",
                               "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized


# _ = rand_float_test(LayerNorm, [2, 4, 768])
# _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post")

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :]  # [batch, position, d_model]
        # visualize_tensor(self.W_E, 'WE')
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed


# rand_int_test(Embed, [2, 4])
# load_gpt2_test(Embed, reference_gpt2.embed, tokens)

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :]  # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed


class OblationInstruction:
    def __init__(self, layer, head_number):
        self.layer = layer
        self.head_number = head_number


# rand_int_test(PosEmbed, [2, 4])
# load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
class Attention(nn.Module):
    def __init__(self, cfg, index, ):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cpu" if M1_MAC else "cuda"))
        self.index = index

    def forward(self, normalized_resid_pre, oblation_instruction=None):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
                   normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_K) + self.b_K

        attn_scores = einsum(
            "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)

        attn_scores = self.apply_causal_mask(attn_scores)
        pattern = attn_scores.softmax(dim=-1)  # [batch, n_head, query_pos, key_pos]

        # if we are instructing oblation then oblate at that layer and head number
        if oblation_instruction:
            o_i = oblation_instruction
            layer = o_i.layer
            head_number = o_i.head_number
            if self.index == layer:
                for row in range(pattern[0][head_number].shape[0]):
                    for column in range(pattern[0][head_number].shape[1]):
                        pattern[0][head_number][row][column] = 0.0

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head",
                   pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z,
                          self.W_O) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device),
                          diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# rand_float_test(Attention, [2, 4, 768])
# load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid,
                     self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out


# rand_float_test(MLP, [2, 4, 768])
# load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])

class TransformerBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg, i)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        self.index = i

    def forward(self, resid_pre, o_i=None):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre,  oblation_instruction=o_i)
        resid_mid = resid_pre + attn_out

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post


# rand_float_test(TransformerBlock, [2, 4, 768])
# load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final,
                        self.W_U) + self.b_U
        return logits


SaveTokensAtPeriodInfo = namedtuple('SaveTokensAtPeriodInfo',['filename', 'end_indexes', 'middle_indexes'])

# rand_float_test(Unembed, [2, 4, 768])
# load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

# class DemoTransformer(nn.Module):
#     def __init__(self, cfg,):
#         super().__init__()
#         self.cfg = cfg
#         self.embed = Embed(cfg)
#         self.pos_embed = PosEmbed(cfg)
#         self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
#         self.ln_final = LayerNorm(cfg)
#         self.unembed = Unembed(cfg)

#     def forward(self, tokens, display=False, save_with_prefix=None, load=False, load_with_mod_vector=None,
#                 intervene_in_resid_at_layer=None, resid_intervention_filename=None, save_tokens_at_index=None,
#                 split_tokens_by_lists=None, split_tokens_by_lists_filename=None, reflect_vector_info=None,
#                 o_i=None):
#         # tokens [batch, position]

#         if load:
#             residual = pickle.load(open('resid.p', 'rb'))

#             plt.plot(residual.detach().numpy().flatten())
#             # plt.show()

#             if load_with_mod_vector is not None:
#                 residual = residual + load_with_mod_vector
#         else:
#             embed = self.embed(tokens)
#             # visualize_tensor(self.embed.W_E, 'we')
#             # print(embed.shape)
#             # visualize_tensor(embed, "Embedding")
#             pos_embed = self.pos_embed(tokens)
#             # print(pos_embed.shape)
#             # visualize_tensor(pos_embed, "Positional Embedding")
#             residual = embed + pos_embed
#             if intervene_in_resid_at_layer == 'start' and resid_intervention_filename:
#                 residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
#                 residual = (residual + torch.from_numpy(residual_intervention)).float()
#             if save_with_prefix:
#                 pickle.dump(residual, open(f'resid_{save_with_prefix}_start.p', 'wb'))
#                 pickle.dump(embed, open(f'resid_{save_with_prefix}_embed.p', 'wb'))
#                 pickle.dump(pos_embed, open(f'resid_{save_with_prefix}_pos_embed.p', 'wb'))







#         # print(residual.shape)
#         for i, block in enumerate(self.blocks):
#             residual = block(residual, o_i)
#             if reflect_vector_info and i == reflect_vector_info['layer']:
#                 cls = reflect_vector_info['cls']
#                 token_index = reflect_vector_info['token_index']
#                 token_as_np_array = residual[0][token_index].detach().numpy()
#                 new_resid_for_token = reflect_vector(token_as_np_array, cls)
#                 # print(sum(sum(sum(residual))))
#                 for j in range(len(residual[0][token_index])):
#                     residual[0][token_index][j] = new_resid_for_token[0][j]
#                 # print(sum(sum(sum(residual))))
#             if i == intervene_in_resid_at_layer and resid_intervention_filename:
#                 residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
#                 print('intervening!')
#                 # print(sum(sum(sum(residual))))
#                 residual = (residual + torch.from_numpy(residual_intervention)).float()
#                 # print(sum(sum(sum(residual))))
#             if save_with_prefix:
#                 pickle.dump(residual, open(f'resid_{save_with_prefix}_{i}.p', 'wb'))
#             if save_tokens_at_index:
#                 filename = save_tokens_at_index.filename
#                 end_indexes = save_tokens_at_index.end_indexes
#                 end_tokens_list = pickle.load(open(f'ends_{filename}_{i}.p', 'rb'))
#                 for index in end_indexes:
#                     end_tokens_list.append(residual[0][index])
#                     # print(f'len end ={len(end_tokens_list)}')
#                 pickle.dump(end_tokens_list, open(f'ends_{filename}_{i}.p', 'wb'))

#                 middle_indexes = save_tokens_at_index.middle_indexes
#                 middle_tokens_list = pickle.load(open(f'middles_{filename}_{i}.p', 'rb'))
#                 for index in middle_indexes:
#                     middle_tokens_list.append(residual[0][index])
#                     # print(f'len middle ={len(middle_tokens_list)}')
#                 pickle.dump(middle_tokens_list, open(f'middles_{filename}_{i}.p', 'wb'))
#             # print(residual)
#             if split_tokens_by_lists and split_tokens_by_lists_filename:
#                 my_filename = split_tokens_by_lists_filename
#                 for key, index_list in split_tokens_by_lists.items():
#                     current_token_list = pickle.load(open(f'{my_filename}_{key}_{i}.p', 'rb'))
#                     for index in index_list:
#                         current_token_list.append(residual[0][index])
#                     pickle.dump(current_token_list, open(f'{my_filename}_{key}_{i}.p', 'wb'))
#         normalized_resid_final = self.ln_final(residual)

#         # visualize_tensor(residual)

#         # pickle.dump(residual, open('resid.p', 'wb'))

#         if display:
#             visualize_tensor(residual, "Residual")
#         #
#         # print(normalized_resid_final)
#         logits = self.unembed(normalized_resid_final)
#         # print(logits)
#         # logits have shape [batch, position, logits]
#         return logits

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens, display=False, save_with_prefix=None, load=False, load_with_mod_vector=None, 
                intervene_in_resid_at_layer=None, resid_intervention_filename=None, extra_value=None,
                intervene_as_fraction_of_resid=None):
        # tokens [batch, position]

        all_resids = []

        if load:
            residual = pickle.load(open('resid.p', 'rb'))

            plt.plot(residual.detach().numpy().flatten())
            #plt.show()

            if load_with_mod_vector is not None:
                residual = residual + load_with_mod_vector
        else:
            embed = self.embed(tokens)
            # visualize_tensor(self.embed.W_E, 'we')
            # print(embed.shape)      
            # visualize_tensor(embed, "Embedding")  
            pos_embed = self.pos_embed(tokens)
            # print(pos_embed.shape)
            # visualize_tensor(pos_embed, "Positional Embedding")
            residual = embed + pos_embed
            if intervene_in_resid_at_layer == 'start' and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                # print('intervening at start!')
                extra_value_array = np.zeros(residual_intervention.shape)
                if extra_value is not None:
                    index1, index2, index3 = extra_value[0]
                    value_to_add = extra_value[1]
                    extra_value_array[index1][index2][index3] += value_to_add
                resid_intervention_factor = 1.0 
                if intervene_as_fraction_of_resid is not None:
                    residual_norm = np.linalg.norm(residual.detach().numpy())
                    resid_intervention_factor = intervene_as_fraction_of_resid * residual_norm / np.linalg.norm(residual_intervention)
                residual = (residual + torch.from_numpy(residual_intervention) * resid_intervention_factor + torch.from_numpy(extra_value_array)).float()
            all_resids.append(residual)
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_start.p', 'wb'))

        # print(residual.shape)
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            if i == intervene_in_resid_at_layer and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                # print('intervening!')
                # print(np.linalg.norm(residual.detach().numpy()))

                resid_intervention_factor = 1.0 
                if intervene_as_fraction_of_resid is not None:
                    residual_norm = np.linalg.norm(residual.detach().numpy())
                    resid_intervention_factor = intervene_as_fraction_of_resid * residual_norm / np.linalg.norm(residual_intervention)            
                
                residual = (residual + torch.from_numpy(residual_intervention) * resid_intervention_factor).float()
            all_resids.append(residual)
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_{i}.p', 'wb'))
            # print(residual)

        normalized_resid_final = self.ln_final(residual)

        # visualize_tensor(residual)

        # pickle.dump(residual, open('resid.p', 'wb'))

        if display:
            visualize_tensor(residual, "Residual")
#
        # print(normalized_resid_final)
        logits = self.unembed(normalized_resid_final)
        # print(logits)
        # logits have shape [batch, position, logits]
        return logits, all_resids


# rand_int_test(DemoTransformer, [2, 4])
# load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

demo_gpt2 = DemoTransformer(get_basic_config(model_name=model_name))
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
# print(cuda(demo_gpt2))


def get_resids_for_sentence(sentence):
    # encoded_sentence = enc.encode(sentence)
    test_tokens = cuda(reference_gpt2.to_tokens(sentence))

    _, all_resids = demo_gpt2(test_tokens)

    return all_resids


def print_top_n_last_token_from_logits(my_logits, n, compare_on_these_token_indices):
    # Get the logits for the last predicted token
    last_logits = my_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()

    # Get the indices of the top n probabilities
    topk_indices = np.argpartition(probabilities, -n)[-n:]
    # Get the top n probabilities
    topk_probabilities = probabilities[topk_indices]
    # Get the top n tokens
    topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

    prob_token_list = list(zip(topk_probabilities, topk_tokens))
    prob_token_list.sort()
    # Print the top n tokens and their probabilities
    for probability, token in prob_token_list:
        print(f"Token: {token}, Probability: {probability}")
    if compare_on_these_token_indices:
        return [probabilities[index] for index in compare_on_these_token_indices]
    else:
        return None


def test_if_token_in_top_n_tokens(my_logits, goal_token, n, print_info=False):
    # Get the logits for the last predicted token
    last_logits = my_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()

    # Get the indices of the top n probabilities
    topk_indices = np.argpartition(probabilities, -n)[-n:]
    # Get the top n probabilities
    topk_probabilities = probabilities[topk_indices]
    # Get the top n tokens
    topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

    prob_token_list = list(zip(topk_probabilities, topk_tokens))
    prob_token_list.sort()
    # Print the top n tokens and their probabilities
    if print_info:
        for probability, token in prob_token_list:
            print(f"Token: {token}, Probability: {probability}")

    goal_token_index = reference_gpt2.tokenizer.encode(goal_token)
    goal_token_prob = probabilities[goal_token_index]
    if print_info:
        print(f"Correct Token ({goal_token}) Probability: {goal_token_prob}")

    if goal_token in topk_tokens:
        return (True, goal_token_prob)
    return (False, goal_token_prob)


def run_gpt2_small_on_string(input_string, prefix, o_i=None, print_logits=False):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    # is enc.encode('?.!') [30, 13, 0]
    end_of_sentence_tokens = [30, 13, 0]
    end_of_sentence_indicies = [ind for ind, ele in enumerate(test_tokens_in[0]) if ele.item() in end_of_sentence_tokens]
    demo_logits_def = demo_gpt2(test_tokens_in, save_with_prefix=prefix, o_i=o_i)
    if print_logits:
        print_top_n_last_token_from_logits(demo_logits_def, 5, None)
    return end_of_sentence_indicies


def check_if_gpt2_gets_right_answer(input_string, correct_next_token):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    demo_logits_def = demo_gpt2(test_tokens_in)
    return test_if_token_in_top_n_tokens(demo_logits_def, correct_next_token, 1)


def get_repeating_token_pattern(cycle_length, total_length):
    random_token_cycle_as_array = [random.choice(LIST_OF_ALL_TOKENS) for _ in range(cycle_length)]
    random_cycle = ''.join(random_token_cycle_as_array)
    num_copies = math.floor(total_length / cycle_length)
    complete_cycles = ''.join([random_cycle for _ in range(num_copies)])
    leftovers = total_length - num_copies*cycle_length
    incomplete_cycle = ''
    next_element = random_token_cycle_as_array[0]
    if leftovers > 0:
        last_elements = random_token_cycle_as_array[:leftovers]
        incomplete_cycle = ''.join(last_elements)
        next_element = random_token_cycle_as_array[leftovers]
    return {
        'next_element': next_element,
        'cycle_array': random_token_cycle_as_array,
        'string': ''.join([complete_cycles, incomplete_cycle])
    }


def get_repeating_token_pattern_and_run_gpt2_small(cycle_length, total_length):
    helper_dict = get_repeating_token_pattern(cycle_length, total_length)
    goal_token = helper_dict['next_element']
    print(f'Goal Token:'+goal_token)
    success = check_if_gpt2_gets_right_answer(helper_dict['string'], goal_token)
    print(success)


def get_angle_between_vectors(a, b):
    dot_product = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return None
    return np.arccos(np.clip(dot_product/norms, -1.0, 1.0))


def get_angle_between_one_d_tensors(a, b):
    a_vec = a.detach().numpy()
    b_vec = b.detach().numpy()
    return get_angle_between_vectors(a_vec, b_vec)


def get_n_random_sentences(n):
    sentences = pickle.load(open('sentences.p', 'rb'))['sentences']
    random_sentence_list = [random.choice(sentences)['sentence'] for _ in range(n)]
    string = " ".join(random_sentence_list)
    return string


def run_gpt_on_n_random_sentences(n, prefix):
    string = get_n_random_sentences(n)
    indicies = run_gpt2_small_on_string(string, prefix)
    return indicies, string


def run_gpt_on_string(string, prefix):
    indicies = run_gpt2_small_on_string(string, prefix)
    return indicies, string


def get_residual_streams_with_prefix(prefix):
    resids = {
        i: pickle.load(open(f'resid_{prefix}_{i}.p', 'rb'))
        for i in range(12)
    }
    resids['start'] = pickle.load(open(f'resid_{prefix}_start.p', 'rb'))
    return resids


def get_angles_from_a_given_index(residual_stream, index):
    print(residual_stream.shape)
    angle_index = [
        (get_angle_between_one_d_tensors(residual_stream[0][index], residual_stream[0][i]),i )
        for i in range(len(residual_stream[0]))
    ]
    angle_index.sort()
    return angle_index


def given_index_get_sorted_angles(residual_streams, index):
    return [
        get_angles_from_a_given_index(residual_stream, index)
        for residual_stream in residual_streams.values()
    ]


def given_index_what_bucket(bucket_ends, index):
    for i in range(len(bucket_ends)):
        if index <= bucket_ends[i]:
            return i
    return len(bucket_ends)


def given_string_end_locations_get_average_vec(residual_streams, sentence_ends):
    number_of_tokens = sentence_ends[-1]+1
    angle_dict = {}
    num_sentences = len(sentence_ends)
    sums = np.zeros((num_sentences, num_sentences))
    counts = np.zeros((num_sentences, num_sentences))
    for key, resid_stream in residual_streams.items():
        angles = np.empty((number_of_tokens, number_of_tokens))
        for i in range(number_of_tokens):
            for j in range(number_of_tokens):
                angles[i][j] = get_angle_between_one_d_tensors(resid_stream[0][i], resid_stream[0][j])

                if i != j:
                    bucket_i = given_index_what_bucket(sentence_ends, i)
                    bucket_j = given_index_what_bucket(sentence_ends, j)
                    sums[bucket_i][bucket_j] += angles[i][j]
                    counts[bucket_i][bucket_j] += 1
        angle_dict[key] = angles
    return angle_dict, sums, counts


# a bullshit function to do what I want
def do_the_thing(n):
    indicies, string = run_gpt_on_n_random_sentences(n)
    residual_dict = get_residual_streams_with_prefix('andrea_messing_around')
    print([residual_dict[11][0][i] - residual_dict[0][0][i] for i in indicies])


def get_random_period_sentence_and_indexes():
    sentence = random.choice(EXAMPLE_SENTENCES)
    test_tokens_in = cuda(reference_gpt2.to_tokens(sentence))
    middle_indicies = [
        ind for ind, ele in enumerate(test_tokens_in[0])
        if ele.item() == 13 and ind != len(test_tokens_in[0])-1
    ]
    return sentence, test_tokens_in, [len(test_tokens_in[0])-1,], middle_indicies


ALL_SENTENCES = EXAMPLE_SENTENCES + EXAMPLE_QUESTIONS


def get_random_period_or_question_sentence_and_indexes():
    sentence = random.choice(ALL_SENTENCES)
    test_tokens_in = cuda(reference_gpt2.to_tokens(sentence))
    middle_indicies = [
        ind for ind, ele in enumerate(test_tokens_in[0])
        if ele.item() == 13 and ind != len(test_tokens_in[0])-1
    ]
    return sentence, test_tokens_in, [len(test_tokens_in[0])-1,], middle_indicies


def get_all_period_or_question_marks_sentence_and_indexes(sentences):
    test_tokens_in = cuda(reference_gpt2.to_tokens(sentences))
    middle_indicies = [
        ind for ind, ele in enumerate(test_tokens_in[0])
        if ele.item() in [30, 13, 0]
    ]
    return sentences, test_tokens_in, middle_indicies


def get_random_period_or_question_sentences_and_indexes(n):
    sentences = [random.choice(ALL_SENTENCES) for _ in range(n)]
    solve_up = [
        get_all_period_or_question_marks_sentence_and_indexes(' '.join(sentences[:i]))
        for i in range(1, n+1)
    ]
    final_sentences, final_test_tokens_in, final_middle_indicies = solve_up[-1]
    end_indicies = [len(test_tokens_in[0])-1 for sentences, test_tokens_in, middle_indicies in solve_up]
    middle_indicies = [i for i in final_middle_indicies if i not in end_indicies]
    assert len(end_indicies) == n
    assert sum([not(final_test_tokens_in[0][i].item() in [30, 13, 0]) for i in end_indicies+middle_indicies]) == 0
    return final_sentences, final_test_tokens_in, end_indicies, middle_indicies


def get_random_sentence_all_other_indicies(n, sentence_list):
    sentences = [random.choice(sentence_list) for _ in range(n)]
    solve_up = [
        get_all_period_or_question_marks_sentence_and_indexes(' '.join(sentences[:i]))
        for i in range(1, n+1)
    ]
    final_sentences, final_test_tokens_in, final_middle_indicies = solve_up[-1]
    end_indicies = [len(test_tokens_in[0])-1 for sentences, test_tokens_in, middle_indicies in solve_up]
    middle_indicies = [j for j in range(len(final_test_tokens_in[0])) if j not in end_indicies]
    assert len(end_indicies) == n
    return final_sentences, final_test_tokens_in, end_indicies, middle_indicies


def compare_internal_vs_external_periods(filename, num_trials, n, sentence_list):
    for _ in tqdm.tqdm(range(num_trials)):
        sentence, test_tokens_in, end_indexes, middle_indexes = get_random_sentence_all_other_indicies(
            n, sentence_list
        )
        info = SaveTokensAtPeriodInfo(filename=filename, end_indexes=end_indexes, middle_indexes=middle_indexes)
        demo_gpt2(test_tokens_in, save_tokens_at_index=info)


def run_and_learn_end_vs_not(filename, num_trials, n):
    for i in range(12):
        pickle.dump([], open(f'ends_{filename}_{i}.p', 'wb'))
        pickle.dump([], open(f'middles_{filename}_{i}.p', 'wb'))

    # split numbers
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.5), n, sentence_list=EXAMPLE_MIX
    )
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.3), n, sentence_list=EXAMPLE_SENTENCES
    )
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.2), n, sentence_list=EXAMPLE_QUESTIONS
    )

    for i in tqdm.tqdm(range(12)):
        learn_you_an_svm(filename, i, "ends", "middles")


def get_random_sentence_punct_last(n, sentence_list):
    sentences = [random.choice(sentence_list) for _ in range(n)]
    solve_up = [
        get_all_period_or_question_marks_sentence_and_indexes(' '.join(sentences[:i]))
        for i in range(1, n+1)
    ]
    final_sentences, final_test_tokens_in, final_middle_indicies = solve_up[-1]
    end_indicies = [len(test_tokens_in[0])-1 for sentences, test_tokens_in, middle_indicies in solve_up]
    middle_indicies = [len(test_tokens_in[0])-2 for sentences, test_tokens_in, middle_indicies in solve_up]
    assert len(end_indicies) == n
    assert len(middle_indicies) == n
    return final_sentences, final_test_tokens_in, end_indicies, middle_indicies


def compare_punct_vs_last_word(filename, num_trials, n, sentence_list):
    for _ in tqdm.tqdm(range(num_trials)):
        sentence, test_tokens_in, word_indexes, last_indexes = get_random_sentence_punct_last(
            n, sentence_list
        )
        info = SaveTokensAtPeriodInfo(filename=filename, end_indexes=word_indexes, middle_indexes=last_indexes)
        demo_gpt2(test_tokens_in, save_tokens_at_index=info)

def run_and_learn_period_vs_last_word(filename, num_trials, n):
    for i in range(12):
        pickle.dump([], open(f'ends_{filename}_{i}.p', 'wb'))
        pickle.dump([], open(f'middles_{filename}_{i}.p', 'wb'))

    # split numbers
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.5), n, sentence_list=EXAMPLE_MIX
    )
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.3), n, sentence_list=EXAMPLE_SENTENCES
    )
    compare_internal_vs_external_periods(
        filename, int(num_trials*0.2), n, sentence_list=EXAMPLE_QUESTIONS
    )

    for i in tqdm.tqdm(range(12)):
        learn_you_an_svm(filename, i, "ends", "middles")


def learn_you_an_svm(filename, i, zero_side, one_side):
    print(len(EXAMPLE_SENTENCES))
    print(len(EXAMPLE_QUESTIONS))
    ends = pickle.load(open(f'{zero_side}_{filename}_{i}.p', 'rb'))
    middles = pickle.load(open(f'{one_side}_{filename}_{i}.p', 'rb'))
    # Convert lists of tensors into 2D arrays
    set1 = np.vstack([t.detach().numpy() for t in ends])
    set2 = np.vstack([t.detach().numpy() for t in middles])

    # Combine the sets into one array for training data
    X = np.vstack((set1, set2))

    # Create labels for the sets. We'll use 0 for set1 and 1 for set2.
    y = np.array([0] * len(set1) + [1] * len(set2))
    print(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SVM Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))

    pickle.dump(clf, open(f'cls_{filename}_{i}.p', 'wb'))
    return clf


def test_sentences_many_lists(file_name, i, cls_file_name):
    test_n_sentences(60, EXAMPLE_SENTENCES, file_name, i, cls_file_name)
    test_n_sentences(20, EXAMPLE_QUESTIONS, file_name, i, cls_file_name)
    test_n_sentences(20, EXAMPLE_MIX, file_name, i, cls_file_name)
    test_n_sentences(5, SEMICOLON_EXAMPLE_SENTENCES, file_name, i, cls_file_name)


def test_n_sentences(n, sen_list, file_name, i, cls_file_name):
    for _ in range(n):
        check_test_against_clf_layer_i(random.choice(sen_list), file_name, i, cls_file_name)


def check_test_against_clf_layer_i(one_sentence, file_name, i, cls_file_name):
    clf = pickle.load(open(f'cls_{cls_file_name}_{i}.p', 'rb'))
    run_gpt2_small_on_string(one_sentence, file_name)
    x = pickle.load(open(f'resid_{file_name}_{i}.p', 'rb'))
    token_vecs = [t.detach().numpy() for t in x[0]]
    set_1 = np.vstack(token_vecs)
    predictions = clf.predict(set_1)
    for j in range(len(predictions)):
        if (j == len(predictions)-1 and predictions[j] == 1) or (j != len(predictions)-1 and predictions[j] == 0):
            print('ERROR')
            tokens_1 = [enc.decode([j]) for j in cuda(reference_gpt2.to_tokens(one_sentence))[0]]
            print([(a, b) for a, b in zip(tokens_1, predictions)])


# Run a test to see how the classifier works
def run_a_test_against_clf_layer_i(string, file_name, i, cls_file_name, prediction_interpreter=None):
    clf = pickle.load(open(f'cls_{cls_file_name}_{i}.p', 'rb'))
    run_gpt2_small_on_string(string, file_name)
    x = pickle.load(open(f'resid_{file_name}_{i}.p', 'rb'))
    token_vecs = [t.detach().numpy() for t in x[0]]
    set_1 = np.vstack(token_vecs)
    predictions = clf.predict(set_1)
    if prediction_interpreter:
        predictions = [prediction_interpreter[prediction] for prediction in predictions]
    tokens_1 = [enc.decode([j]) for j in cuda(reference_gpt2.to_tokens(string))[0]]
    return [(a, b) for a, b in zip(tokens_1, predictions)]


def get_random_sentence_breakdown_token_split(n, breakdowns, keys):

    sentence_breakdowns = [random.choice(breakdowns) for _ in range(n)]
    print(sentence_breakdowns)
    token_index_lists = {
        key: []
        for key in keys
    }

    flattened_sentence_breakdowns = [item for sublist in sentence_breakdowns for item in sublist]
    print(flattened_sentence_breakdowns)

    old_string = ''
    old_tokenization = cuda(reference_gpt2.to_tokens(old_string))

    for breakdown_part in flattened_sentence_breakdowns:
        new_string_fragment = \
            breakdown_part['fragment'] \
                if (i==0 or breakdown_part['type'] in ['end', 'end of sentence']) \
                else " "+breakdown_part['fragment']
        new_string = old_string + new_string_fragment
        new_tokenization = cuda(reference_gpt2.to_tokens(new_string))
        # print(f'new_string={new_string} | old_string={old_string}')
        # print(f'new_tokenization={new_tokenization} | old_tokenization={old_tokenization}')

        # We assume all new tokens get the label of the last part we added
        # note that this can be bullshit "Calif." tokenizes the period as 13 and "Calif.?" tokenizes
        # the .? together
        new_token_indexes = list(range(len(old_tokenization[0]), len(new_tokenization[0])))
        print(new_token_indexes)
        new_list = token_index_lists[breakdown_part['type']] + new_token_indexes
        print(new_list)
        token_index_lists[breakdown_part['type']] = new_list

        old_string = new_string
        old_tokenization = new_tokenization

    # so by the end we have added on all the indices of the tokenzation.
    print(new_string)
    print(new_tokenization)
    print(token_index_lists)

    return new_tokenization, token_index_lists


def run_and_learn_subject_vs_end_vs_middle(filename, num_trials, n):
    updated_filename = 'split_token_by_lists_'+filename
    for i in range(12):
        for name in EXAMPLE_SUBJECT_BREAKDOWNS_KEYS:
            pickle.dump([], open(f'{updated_filename}_{name}_{i}.p', 'wb'))

    # split numbers
    for _ in tqdm.tqdm(range(num_trials)):
        tokenization, token_index_lists = get_random_sentence_breakdown_token_split(
            random.randint(1, n),
            EXAMPLE_SUBJECT_BREAKDOWNS,
            EXAMPLE_SUBJECT_BREAKDOWNS_KEYS,
        )
        demo_gpt2(
            tokenization,
            split_tokens_by_lists=token_index_lists,
            split_tokens_by_lists_filename=updated_filename
        )

    for i in range(12):
        learn_you_a_multi_part_svm(updated_filename, i, EXAMPLE_SUBJECT_BREAKDOWNS_KEYS)


def run_and_learn_verb_subject_vs_end_vs_other(filename, num_trials, n):
    upload_filename = 'andrea_verbing_'+filename
    # create and or empty out the files
    for i in range(12):
        for name in EXAMPLE_SUBJECT_VERB_BREAKDOWNS_KEYS:
            pickle.dump([], open(f'{upload_filename}_{name}_{i}.p', 'wb'))

    for _ in tqdm.tqdm(range(num_trials)):
        tokenization, token_index_lists = get_random_sentence_breakdown_token_split(
            random.randint(1, n),
            EXAMPLE_SUBJECT_VERB_BREAKDOWNS,
            EXAMPLE_SUBJECT_VERB_BREAKDOWNS_KEYS,
        )
        demo_gpt2(
            tokenization,
            split_tokens_by_lists=token_index_lists,
            split_tokens_by_lists_filename=upload_filename
        )

    for i in range(12):
        learn_you_a_multi_part_svm(upload_filename, i, EXAMPLE_SUBJECT_VERB_BREAKDOWNS_KEYS)


def learn_you_a_multi_part_svm(filename, i, part_names):
    token_sets = [pickle.load(open(f'{filename}_{part_name}_{i}.p', 'rb')) for part_name in part_names]
    # Convert lists of tensors into 2D arrays
    numpy_sets = [np.vstack([t.detach().numpy() for t in token_set]) for token_set in token_sets]

    # Combine the sets into one array for training data
    X = np.vstack(tuple(numpy_sets))

    print(f'Len X= {len(X)} on layer {i}')

    # Create labels for the sets. We'll use 0 for set1 and 1 for set2.
    y = np.array([j for j, token_set in enumerate(token_sets) for _ in token_set])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SVM Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))

    pickle.dump(clf, open(f'cls_{filename}_{i}.p', 'wb'))
    return clf


def demo_of_subject_end_of_sentence_other(sentence, layer):
    test_string = "test_andrea"
    file_name = "split_token_by_lists_test_subject"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "Subject",
            1: "Other",
            2:"End of Sentence"
        }
    )


def oblate_and_run_a_test_against_clf_layer_i(string, file_name, i, cls_file_name,
                                              prediction_interpreter=None,
                                              o_i=None,
                                              ):
    clf = pickle.load(open(f'cls_{cls_file_name}_{i}.p', 'rb'))
    run_gpt2_small_on_string(string, file_name,o_i, True)
    x = pickle.load(open(f'resid_{file_name}_{i}.p', 'rb'))
    token_vecs = [t.detach().numpy() for t in x[0]]
    set_1 = np.vstack(token_vecs)
    predictions = clf.predict(set_1)
    if prediction_interpreter:
        predictions = [prediction_interpreter[prediction] for prediction in predictions]
    tokens_1 = [enc.decode([j]) for j in cuda(reference_gpt2.to_tokens(string))[0]]
    return [(a, b) for a, b in zip(tokens_1, predictions)]



def oblate_and_run(sentence):
    test_string = "test_andrea"
    file_name = "middle_vs_not"


    for layer in range(12):
        for head_number in range(12):
            o_i = OblationInstruction(layer, head_number)
            origional = run_a_test_against_clf_layer_i(
                sentence, test_string, 11, file_name,
                prediction_interpreter={
                    0: "End of Sentence",
                    1: "Other",
                }
            )

            new = oblate_and_run_a_test_against_clf_layer_i(
                sentence, test_string, 11, file_name,
                prediction_interpreter={
                    0: "End of Sentence",
                    1: "Other",
                },
                o_i=o_i,
            )

            if origional != new:
                print(f'===== DIFF at Layer={layer} and head_number={head_number}')
                print(list(zip(origional,new)))
                print(f'<<<<<<<<<<<<Differences >>>>>>>>>>>>>>>>>>>')
                for orig_tok, new_tok in zip(origional,new):
                    if orig_tok != new_tok:
                        print(f'orig={orig_tok} new={new_tok}')

                print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

def demo_of_end_of_sentence_vs_other(sentence, layer):
    test_string = "test_andrea"
    file_name = "middle_vs_not"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "End of Sentence",
            1: "Other",
        }
    )


def demo_of_verbing_verbs(sentence, layer):
    test_string = "test_andrea"
    file_name = "andrea_verbing_verbs"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "Subject",
            1: "Verb",
            2: "Other",
            3: "End of Sentence"
        }
    )


def get_undo_layer_i(file_name, i):
    cls = pickle.load(open(f'cls_{file_name}_{i}.p', 'rb'))
    w = cls.coef_[0]
    w_norm = w / np.linalg.norm(w)
    return w_norm


def make_intervention_on_one_token(change_index, length, file_name, layer_number, multiplier):
    w_norm = multiplier * get_undo_layer_i(file_name, layer_number)
    array = np.zeros((1, length, 768))
    array[0, change_index, :] = w_norm
    pickle.dump(array, open(f'w_norm{file_name}_index_{change_index}_layer_{layer_number}.p', 'wb'))


def reflect_vector(X, clf):
    X = X.reshape(1, -1)  # reshape the data
    # Calculate the distance of the point to the hyperplane
    distance = np.abs(clf.decision_function(X)) / np.linalg.norm(clf.coef_)

    # Calculate the direction to move the point
    direction = np.sign(clf.decision_function(X))

    # Reflect the point across the hyperplane
    X_reflected = X - 2 * distance * direction * (clf.coef_ / np.linalg.norm(clf.coef_))

    # Predict the classes of X and X_reflected
    y_pred = clf.predict(X)
    y_pred_reflected = clf.predict(X_reflected)

    print(f'Predicted class for X: {y_pred[0]}')
    print(f'Predicted class for X_reflected: {y_pred_reflected[0]}')

    return X_reflected


def run_gpt_demo_with_intervention(input_string, layer, filename):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    period_index = len(test_tokens_in)-1

    print("======= DEFAULT =========")
    default_logits = demo_gpt2(test_tokens_in,)
    print_top_n_last_token_from_logits(default_logits, 5, None)

    print(f"======= REFLECTING =========")
    my_logits = demo_gpt2(
        test_tokens_in,
        f={
            'layer': layer,
            'cls': pickle.load(open(f'cls_{filename}_{layer}.p', 'rb')),
            'token_index': period_index,
        },
    )
    print_top_n_last_token_from_logits(my_logits, 5, None)

    for multiplier in [-1000, -100, -20, 20, 100, 1000]:
        print(f"======= WITH INTERVENTION {multiplier}=========")
        make_intervention_on_one_token(period_index, len(test_tokens_in), filename, layer, multiplier)
        my_logits = demo_gpt2(
            test_tokens_in,
            intervene_in_resid_at_layer=layer,
            resid_intervention_filename=f'w_norm{filename}_index_{period_index}_layer_{layer}.p',
        )
        print_top_n_last_token_from_logits(my_logits, 5, None)


SEN_COMP_TOKEN_LENGTH = 9
SEN_COMP_CLF_FILE_NAME = "sentence_completeness_clf_layer"
SEN_DATA_SET_FILE_NAME = "eight_token_sentences.p"
SEN_ID_TO_COMPLETENESS_FILE_NAME = "eight_token_index_to_completeness.p"
SEN_ID_TO_SEN_FILE_NAME = "eight_token_index_to_sentence.p"


def get_complete_sentences():
    sen_id_to_sen = pickle.load(open(SEN_ID_TO_SEN_FILE_NAME, 'rb'))
    sen_id_to_comp = pickle.load(open(SEN_ID_TO_COMPLETENESS_FILE_NAME, 'rb'))
    return [sen_id_to_sen[key] for key, value in sen_id_to_comp.items() if value == 1]

def get_sentence_completeness_clf(layer):
    if not layer in range(11):
        raise Exception("We have no classifier for the requested layer number")

    return pickle.load(open(f'{SEN_COMP_CLF_FILE_NAME}_{layer}.p', 'rb'))


def check_if_gpt2_gets_right_answer_ablation(input_string, correct_next_token, oblation_instruction, prefix, print_info=False):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    demo_logits_def = demo_gpt2(test_tokens_in, save_with_prefix=prefix, o_i=oblation_instruction)
    # (bool, prob)
    return test_if_token_in_top_n_tokens(demo_logits_def, correct_next_token, 5, print_info=print_info)


# TODO: get the file name
SEN_COMP_MEAN_RESID_FILE_NAME = "mean_vector_layer"


def get_completeness_mean(layer):
    # mean_vector_layer_0.p
    return pickle.load(open(f'{SEN_COMP_MEAN_RESID_FILE_NAME}_{layer}.p', 'rb'))


def oblate_and_run_a_completeness_test_against_clf_layer_i(string, prefix, oblation_instruction=None):
    right_answer = check_if_gpt2_gets_right_answer_ablation(string, '.', oblation_instruction, prefix)

    clf_answers_layer_by_layer = []
    for layer in [10,]:
        x = pickle.load(open(f'resid_{prefix}_{layer}.p', 'rb'))
        clf = get_sentence_completeness_clf(layer)
        # TODO: use clf for something
        vec_resid = x.detach().numpy().flatten()
        assert len(vec_resid) == 6912
        vec_minus_mean = vec_resid - get_completeness_mean(layer)
        clf_answers_layer_by_layer.append(
            (layer, clf.predict(
                np.array([vec_minus_mean,])
            )
             )
        )

    return right_answer, clf_answers_layer_by_layer


def sen_completeness_oblate_and_run(sentence, diff):
    test_string = "test_andrea"

    origional_answer, origional_clf_answers = oblate_and_run_a_completeness_test_against_clf_layer_i(
        sentence, test_string,)

    push_up = []
    push_down = []

    print(origional_answer, origional_clf_answers)

    for layer in range(12):
        for head_number in range(12):
            o_i = OblationInstruction(layer, head_number)
            new_answer, new_clf_answers = oblate_and_run_a_completeness_test_against_clf_layer_i(
                sentence, test_string,
                oblation_instruction=o_i,
            )

            difference = new_answer[1] - origional_answer[1]

            if  difference > diff:
                #print(f"Layer Number={layer} | Head Number={head_number}")
                #print(new_answer, new_clf_answers)
                push_up.append(
                    ((layer, head_number), difference)
                )

            if new_answer[1] - origional_answer[1] < -diff:
                #print(f"Layer Number={layer} | Head Number={head_number}")
                #print(new_answer, new_clf_answers)
                push_down.append(
                    ((layer, head_number),difference)
                )

    return push_up, push_down



def run_on_many_sentences(sentences, diff):
    all_ups = []
    all_downs = []
    for sentence in sentences:
        push_up, push_down = sen_completeness_oblate_and_run(sentence, diff)
        all_ups.append(push_up)
        all_downs.append(push_down)

    layer_impacts_up = {}
    layer_impacts_down = {}

    for array in all_ups:
        for intervention_name, difference in array:
            layer_impacts_up[intervention_name] = layer_impacts_up.get(intervention_name, 0) + difference

    for array in all_downs:
        for intervention_name, difference in array:
            layer_impacts_down[intervention_name] = layer_impacts_down.get(intervention_name, 0) + difference

    return all_ups, all_downs, layer_impacts_up, layer_impacts_down








# IPython.embed()
