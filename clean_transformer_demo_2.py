# try:
#     import google.colab
#     IN_COLAB = True
#     print("Running as a Colab notebook")
#   %pip install git+https://github.com/neelnanda-io/Easy-Transformer.git@clean-transformer-demo
  # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs  
#   %pip install git+https://github.com/neelnanda-io/PySvelte.git
#   %pip install fancy_einsum
#   %pip install einops
# except:
IN_COLAB = False
M1_MAC = True

# print("Running as a Jupyter notebook - intended for development only!")

import einops
from fancy_einsum import einsum
from dataclasses import dataclass
# from easy_transformer import EasyTransformer
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from transformer_lens.utils import get_corner, gelu_new, tokenize_and_concatenate
# from easy_transformer import loading_from_pretrained as loading
# from easy_transformer.loading_from_pretrained import function_or_class_you_want as loading
from transformer_lens import loading_from_pretrained as loading


from time import sleep

import tqdm.auto as tqdm
import matplotlib.pyplot as plt
from typing import Optional
import string
import random


def alphabetic_rank(word):
    base = ord('a')
    rank = 0
    current_letter_value = 1
    for char in word.lower():
        if char.isalpha():
            rank += current_letter_value * (ord(char) - base + 1)
        else:
            raise ValueError(f"Non-alphabetic character '{char}' found in word")
        
        current_letter_value *= 1/26
    return rank


# import IPython
# IPython.embed()

def visualize_tensor(tensor, title, dim=0, cmap="viridis", max_cols=5, scaling_factor=4):
    tensor = tensor.detach().cpu().numpy()
    num_slices = tensor.shape[dim]
    num_rows = int(np.ceil(num_slices / max_cols))
    num_cols = min(num_slices, max_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(scaling_factor * max_cols, scaling_factor * num_rows), constrained_layout=True)

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

    #plt.show()

# reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

# model_name = "gpt2-large"
model_name = "gpt2-small"
# model_name = "pythia-70m-v0"

reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False, center_writing_weights=False)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n:n[1])
# print(sorted_vocab[:20])
# print()
# print(sorted_vocab[250:270])
# print()
# print(sorted_vocab[990:1010])
# print()

# print(sorted_vocab[-20:])

# print(reference_gpt2.to_tokens("Whether a word begins with a capital or space matters!"))
# print(reference_gpt2.to_tokens("Whether a word begins with a capital or space matters!", prepend_bos=False))

# print(reference_gpt2.to_str_tokens("Ralph"))
# print(reference_gpt2.to_str_tokens(" Ralph"))
# print(reference_gpt2.to_str_tokens(" ralph"))
# print(reference_gpt2.to_str_tokens("ralph"))

# print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
# print(tokens)
# print(tokens.shape)
# print(reference_gpt2.to_str_tokens(tokens))

def cuda(x):
    return x.to('cpu') if M1_MAC else x.cuda()

tokens = cuda(tokens)
logits, cache = reference_gpt2.run_with_cache(tokens)
# print(logits.shape)

log_probs = logits.log_softmax(dim=-1)
probs = logits.log_softmax(dim=-1)
# print(log_probs.shape)
# print(probs.shape)

# print(list(zip(reference_gpt2.to_str_tokens(reference_text), reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0]))))

next_token = logits[0, -1].argmax(dim=-1)
# print(next_token)

next_tokens = torch.cat([tokens, torch.tensor(next_token, device='cpu' if M1_MAC else 'cuda', dtype=torch.int64)[None, None]], dim=-1)
new_logits = reference_gpt2(next_tokens)
# print("New Input:", next_tokens)
# print(next_tokens.shape)
# print("New Input:", reference_gpt2.tokenizer.decode(next_tokens[0]))

# print(new_logits.shape)
# print(new_logits[-1, -1].argmax(-1))

# print(reference_gpt2.tokenizer.decode(new_logits[-1, -1].argmax(-1)))

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

# cfg = Config()
# print(cfg)

# import IPython
# IPython.embed()

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

print(loading.get_pretrained_model_config(model_name))
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randn(shape))
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randint(100, 1000, shape))
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
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
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
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
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + cfg.layer_norm_eps).sqrt()
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
        embed = self.W_E[tokens, :] # [batch, position, d_model]
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
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed

# rand_int_test(PosEmbed, [2, 4])
# load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

class Attention(nn.Module):
    def __init__(self, cfg):
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
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)
        
        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
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
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

# rand_float_test(MLP, [2, 4, 768])
# load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
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
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits

# rand_float_test(Unembed, [2, 4, 768])
# load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
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
                    assert np.isclose(np.linalg.norm(residual_intervention), 1.0)
                    residual_norm = np.linalg.norm(residual.detach().numpy())
                    resid_intervention_factor = intervene_as_fraction_of_resid * residual_norm
                residual = (residual + torch.from_numpy(residual_intervention) * resid_intervention_factor + torch.from_numpy(extra_value_array)).float()
            all_resids.append(residual)
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_start.p', 'wb'))

        # print(residual.shape)
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            if i == intervene_in_resid_at_layer and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                print('intervening!')
                print(np.linalg.norm(residual.detach().numpy()))

                resid_intervention_factor = 1.0 
                if intervene_as_fraction_of_resid is not None:
                    assert np.isclose(np.linalg.norm(residual_intervention), 1.0)
                    residual_norm = np.linalg.norm(residual.detach().numpy())
                    resid_intervention_factor = intervene_as_fraction_of_resid * residual_norm                
                
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
print(cuda(demo_gpt2))

# test_string = """Mini scule is a species of microhylid frog endemic to Madagascar that was described in 2019. The scientific name of the species refers to its size, being a pun on the word minuscule. It is very small, measuring only 8.4 to 10.8 mm (0.33 to 0.43 in) in snout–vent length. It has bronze underparts with a brown groin and back of the thigh, cream upperparts with brown flecking, a dark brown side of the head, and a red iris. On the hind feet, the first toe is absent and the second and fifth toes are strongly reduced. The frog is known only from the Sainte Luce Reserve, where it inhabits areas with deep leaf litter near semi-permanent water bodies. Specimens of frogs from Mandena, the Vohimena mountains, the southern Anosy Mountains, and Tsitongambarika may also be of this species. Along with Mini mum and Mini ature, the other two species in its genus, it received media attention when first described due to the wordplay in its scientific name. (Full article...)"""

# test_tokens = cuda(reference_gpt2.to_tokens(test_string))
# demo_logits = demo_gpt2(test_tokens)

# def lm_cross_entropy_loss(logits, tokens):
#     # Measure next token loss
#     # Logits have shape [batch, position, d_vocab]
#     # Tokens have shape [batch, position]
#     log_probs = logits.log_softmax(dim=-1)
#     pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
#     return -pred_log_probs.mean()
# loss = lm_cross_entropy_loss(demo_logits, test_tokens)
# print(loss)
# print("Loss as average prob", (-loss).exp())
# print("Loss as 'uniform over this many variables'", (loss).exp())
# print("Uniform loss over the vocab", math.log(demo_gpt2.cfg.d_vocab))

if False:
    import pickle
    import IPython
    WEIRD_TOKENS = pickle.load(open('weird_tokens_int.p', 'rb'))
    # IPython.embed()
    def argmin_excluding_weird_tokens(tensor):
        exclude_indices = WEIRD_TOKENS

        # Create a mask for the elements to exclude
        mask = torch.ones_like(tensor, dtype=torch.bool)
        mask[exclude_indices] = False

        # Set the masked elements to a large value
        large_value = torch.finfo(tensor.dtype).max
        tensor_excluded = tensor.clone()
        tensor_excluded[mask == False] = large_value

        # Compute the argmin of the resulting tensor
        argmin_excluded = torch.argmin(tensor_excluded)

        return argmin_excluded


    def generate_random_string(length):
        # Define the set of characters to use (alphabetic characters and spaces)
        characters = string.ascii_letters + '         '

        # Generate a random string of the specified length
        random_string = ''.join(random.choice(characters) for _ in range(length))

        return random_string


    vocab_size = 50257
    running_probabilities = np.array([0.] * vocab_size)


    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"runnersllerhipISAnutsBILITYouls\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \" o2efnisdnp2e23-s\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"disclaimeruceStarsativelyitionallyports\". Reply: \""
    # test_string = f"Hello my name is Adam and I"
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"Barack Obama\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"Alex Arkhipov\". Reply: \""
    # test_string = "When Tamarale and Tamastate went to the store, Tamarale gave a drink to"
    # test_string = "When Tamarale and Tamastate went to the store, Tamastate gave a drink to"
    # test_string = "When Tamastate and Tamarale went to the store, Tamarale gave a drink to"
    # test_string = "When Tamastate and Tamarale went to the store, Tamastate gave a drink to"
    # test_string = "When Guinevere and Romanova went to the store, Guinevere gave a drink to"
    # test_string = "When Romanova and Colberta went to the store, Romanova gave a drink to"
    # test_string = "When Tweedle Dee and Tweedle Dum went to the store, Tweedle Dum gave a drink to"
    # test_string = "When Alexandria-Solstice and Alexandria-Equinox went to the park, Alexandria-Solstice bought ice cream for Alex"
    # test_string = "1. e4 e5 2. Nf3 Nc6 3. Bc4"
    # test_string = " pourtant je ne sais pas pourquoi "
    test_string = " He didn't understand, so I told"

    # resid = pickle.load(open('resid.p', 'rb'))
    # resid_zeros = torch.zeros(resid.shape)

    # print(resid_zeros[0][0][0])
    # resid_zeros[0][0][0] -= 1.

    # # visualize_tensor(resid_zeros, 'hello')

    # def get_output(mod_vector):
    #     demo_logits = demo_gpt2('abcd', load=True, load_with_mod_vector = mod_vector)

    #     return reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())



    # print(get_output(resid_zeros))
    # raise Exception()

    for i in tqdm.tqdm(range(5)):
        test_tokens = cuda(reference_gpt2.to_tokens(test_string))
        print(reference_gpt2.to_str_tokens(test_tokens))

        demo_logits = demo_gpt2(test_tokens)

        # Get the logits for the last predicted token
        last_logits = demo_logits[-1, -1]
        # Apply softmax to convert the logits to probabilities
        probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
        
        # Get the indices of the top 10 probabilities
        topk_indices = np.argpartition(probabilities, -10)[-10:]
        # Get the top 10 probabilities
        topk_probabilities = probabilities[topk_indices]
        # Get the top 10 tokens
        topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

        # Print the top 10 tokens and their probabilities
        for token, probability in zip(topk_tokens, topk_probabilities):
            print(f"Token: {token}, Probability: {probability}")

        # Get the most probable next token and append it to the test string
        most_probable_next_token = reference_gpt2.tokenizer.decode(last_logits.argmax())

        test_string += most_probable_next_token

        # if i == 0:
        #     test_string += ' Tam'
        # else:
        #     test_string += most_probable_next_token

    # for i in tqdm.tqdm(range(5)):
    #     test_tokens = cuda(reference_gpt2.to_tokens(test_string))
    #     print(reference_gpt2.to_str_tokens(test_tokens))

    #     demo_logits = demo_gpt2(test_tokens)#, display=i==0, load=i==0, load_with_mod_vector = None)
    #     print(demo_logits[-1, -1].shape)
        
    #     logits = demo_logits[-1, -1].detach().numpy()
    #     # running_probabilities = np.multiply(running_probabilities, logits)
    #     # running_probabilities = logits
    #     # adjusted_probabilities = logits - running_probabilities
    #     # running_probabilities = running_probabilities * i / (i + 1) + logits / (i + 1)
    #     # running_probabilities = logits

    #     # test_string += reference_gpt2.tokenizer.decode(argmin_excluding_weird_tokens(torch.from_numpy(adjusted_probabilities)))

    #     # print(demo_logits[-1, -1][demo_logits[-1, -1].argmax()])



    #     test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    #     # test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmin())
    #     # test_string += reference_gpt2.tokenizer.decode(argmin_excluding_weird_tokens(demo_logits[-1, -1]))

    print(test_string)
    raise Exception()
    
import os

def file_exists(filepath):
    return os.path.isfile(filepath)


if False:
    import pickle
    import tiktoken

    sentences = pickle.load(open('sentences.p', 'rb'))['sentences']

    enc = tiktoken.get_encoding('r50k_base')
    def pad_sentence_with_newlines_or_reject(sentence, up_to_token_length):
        sentence_token_length = len(enc.encode(sentence))
        if sentence_token_length > up_to_token_length:
            return None
        padded_sentence = ''.join(['\n\n'] * (up_to_token_length - sentence_token_length)) + sentence
        return_sentence = padded_sentence[2:] if len(enc.encode(padded_sentence)) == up_to_token_length + 1 else f'\n\n{padded_sentence}' if len(enc.encode(padded_sentence)) == up_to_token_length - 1 else padded_sentence
        # print(padded_sentence)
        # print(len(enc.encode(return_sentence)))
        # print([return_sentence])
        if len(enc.encode(return_sentence)) == up_to_token_length:
            return return_sentence
        return None
    

    from collections import defaultdict

    male_path = 'male_resid_streams.p'
    female_path = 'female_resid_streams.p'
    all_path = 'all_resid_streams.p'
    index_to_sentence_path = 'index_to_sentence.p'

    # male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    # female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open(index_to_sentence_path, 'rb')) if file_exists(index_to_sentence_path) else {}



    def add_streams_to_storage(streams, stream_storage):
        stream_storage['start'].append(streams[0])
        for i, stream in enumerate(streams[1:]):
            stream_storage[i].append(stream)


    # seen_last_halt_point = False

    for i, sentence_dict in enumerate(sentences):
        # if (subject := sentence.get('subject')) and (gender := subject.get('gender')) == 'female':
        raw_sentence = sentence_dict['sentence']

        if i < 1300:
            continue

        sentence = pad_sentence_with_newlines_or_reject(raw_sentence, 12)

        if sentence:
            # if 'Patrick sat gazing into space' in sentence:
            #     seen_last_halt_point = True            
            # if not seen_last_halt_point:
            #     continue
            subject = sentence_dict.get('subject')
            if subject:
                gender = subject.get('gender')
            
                test_tokens = cuda(reference_gpt2.to_tokens(sentence))
                print(reference_gpt2.to_str_tokens(test_tokens))
                demo_logits, resid_streams = demo_gpt2(test_tokens)

                # if gender == 'male':
                #     add_streams_to_storage(resid_streams, male_resid_streams)
                #     add_streams_to_storage(resid_streams, all_resid_streams)
                    
                # elif gender == 'female':
                #     add_streams_to_storage(resid_streams, female_resid_streams)
                #     add_streams_to_storage(resid_streams, all_resid_streams)                    

                add_streams_to_storage(resid_streams, all_resid_streams)

                index_to_sentence[len(all_resid_streams[0]) - 1] = sentence

        if i % 50 == 0:
            print(i)
            # pickle.dump(male_resid_streams, open(male_path, 'wb'))
            # pickle.dump(female_resid_streams, open(female_path, 'wb'))
            pickle.dump(all_resid_streams, open(all_path, 'wb'))
            pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))
    # pickle.dump(male_resid_streams, open(male_path, 'wb'))
    # pickle.dump(female_resid_streams, open(female_path, 'wb'))
    pickle.dump(all_resid_streams, open(all_path, 'wb'))
    pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))




    raise Exception()


if False:
    import pickle
    import tiktoken
    from collections import defaultdict

    sentences = pickle.load(open('eight_token_sentences.p', 'rb'))

    all_path = 'eight_token_resid_streams.p'
    index_to_sentence_path = 'eight_token_index_to_sentence.p'

    # male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    # female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open(index_to_sentence_path, 'rb')) if file_exists(index_to_sentence_path) else {}



    def add_streams_to_storage(streams, stream_storage):
        stream_storage['start'].append(streams[0])
        for i, stream in enumerate(streams[1:]):
            stream_storage[i].append(stream)


    # seen_last_halt_point = False

    for i, sentence in enumerate(sentences):

        if i < 2800:
            continue


        # if 'Patrick sat gazing into space' in sentence:
        #     seen_last_halt_point = True            
        # if not seen_last_halt_point:
        #     continue
        test_tokens = cuda(reference_gpt2.to_tokens(sentence))
        print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits, resid_streams = demo_gpt2(test_tokens)        

        add_streams_to_storage(resid_streams, all_resid_streams)

        index_to_sentence[len(all_resid_streams[0]) - 1] = sentence

        if i % 50 == 0:
            print(i)
            # pickle.dump(male_resid_streams, open(male_path, 'wb'))
            # pickle.dump(female_resid_streams, open(female_path, 'wb'))
            pickle.dump(all_resid_streams, open(all_path, 'wb'))
            pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))
    # pickle.dump(male_resid_streams, open(male_path, 'wb'))
    # pickle.dump(female_resid_streams, open(female_path, 'wb'))
    pickle.dump(all_resid_streams, open(all_path, 'wb'))
    pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))




    raise Exception()


if True:
    import pickle
    import tiktoken
    from collections import defaultdict

    sentences = pickle.load(open('10_token_sentences.p', 'rb'))

    all_path = '10_token_resid_streams.p'
    index_to_sentence_path = '10_index_to_sentence.p'

    # male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    # female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open(index_to_sentence_path, 'rb')) if file_exists(index_to_sentence_path) else {}



    def add_streams_to_storage(streams, stream_storage):
        stream_storage['start'].append(streams[0])
        for i, stream in enumerate(streams[1:]):
            stream_storage[i].append(stream)


    # seen_last_halt_point = False

    import sys
    begin_at = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    for i, sentence in enumerate(sentences):

        if i < begin_at:
            continue

        if i > begin_at + 325:
            raise Exception()


        # if 'Patrick sat gazing into space' in sentence:
        #     seen_last_halt_point = True            
        # if not seen_last_halt_point:
        #     continue
        test_tokens = cuda(reference_gpt2.to_tokens(sentence))
        print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits, resid_streams = demo_gpt2(test_tokens)        

        add_streams_to_storage(resid_streams, all_resid_streams)

        index_to_sentence[len(all_resid_streams[0]) - 1] = sentence

        if i % 50 == 0:
            print(i)
            # pickle.dump(male_resid_streams, open(male_path, 'wb'))
            # pickle.dump(female_resid_streams, open(female_path, 'wb'))
            pickle.dump(all_resid_streams, open(all_path, 'wb'))
            pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))
    # pickle.dump(male_resid_streams, open(male_path, 'wb'))
    # pickle.dump(female_resid_streams, open(female_path, 'wb'))
    pickle.dump(all_resid_streams, open(all_path, 'wb'))
    pickle.dump(index_to_sentence, open(index_to_sentence_path, 'wb'))




    raise Exception()



if False:
    import pickle
    import tiktoken
    from collections import defaultdict

    # sentences = pickle.load(open('eight_token_sentences.p', 'rb'))

    # all_path = 'eight_token_resid_streams.p'
    index_to_sentence_path = 'eight_token_index_to_sentence.p'
    index_to_completeness_path = 'eight_token_index_to_completeness.p'

    # male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    # female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    # all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open(index_to_sentence_path, 'rb')) if file_exists(index_to_sentence_path) else {}
    index_to_completeness = pickle.load(open(index_to_completeness_path, 'rb')) if file_exists(index_to_completeness_path) else {}


    # def add_streams_to_storage(streams, stream_storage):
    #     stream_storage['start'].append(streams[0])
    #     for i, stream in enumerate(streams[1:]):
    #         stream_storage[i].append(stream)


    # seen_last_halt_point = False

    for i, sentence in index_to_sentence.items():

        # if i < 2800:
        #     continue


        # if 'Patrick sat gazing into space' in sentence:
        #     seen_last_halt_point = True            
        # if not seen_last_halt_point:
        #     continue
        test_tokens = cuda(reference_gpt2.to_tokens(sentence))
        # print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits, _ = demo_gpt2(test_tokens)        

        probabilities = torch.nn.functional.softmax(demo_logits[-1,-1], dim=0).detach().numpy()

        if probabilities[13] > 0.05:
            # Is or could be a complete sentence
            index_to_completeness[i] = 1
        else:
            index_to_completeness[i] = 0

        print(sentence)
        print(probabilities)
        print(probabilities.shape)
        print(probabilities[13])

        # add_streams_to_storage(resid_streams, all_resid_streams)

        # index_to_sentence[len(all_resid_streams[0]) - 1] = sentence

        if i % 50 == 0:
            print(i)
            # # pickle.dump(male_resid_streams, open(male_path, 'wb'))
            # # pickle.dump(female_resid_streams, open(female_path, 'wb'))
            # pickle.dump(all_resid_streams, open(all_path, 'wb'))
            pickle.dump(index_to_completeness, open(index_to_completeness_path, 'wb'))
    # pickle.dump(male_resid_streams, open(male_path, 'wb'))
    # pickle.dump(female_resid_streams, open(female_path, 'wb'))
    # pickle.dump(all_resid_streams, open(all_path, 'wb'))
    pickle.dump(index_to_completeness, open(index_to_completeness_path, 'wb'))



if False:
    import pickle
    import tiktoken
    from collections import defaultdict

    # sentences = pickle.load(open('eight_token_sentences.p', 'rb'))

    # all_path = 'eight_token_resid_streams.p'
    index_to_sentence_path = 'eight_token_index_to_sentence.p'
    index_to_completeness_path = 'eight_token_index_to_completeness.p'

    # male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    # female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    # all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open(index_to_sentence_path, 'rb')) if file_exists(index_to_sentence_path) else {}
    index_to_completeness = pickle.load(open(index_to_completeness_path, 'rb')) if file_exists(index_to_completeness_path) else {}


    # def add_streams_to_storage(streams, stream_storage):
    #     stream_storage['start'].append(streams[0])
    #     for i, stream in enumerate(streams[1:]):
    #         stream_storage[i].append(stream)


    # seen_last_halt_point = False

    for i, sentence in index_to_sentence.items():

        # if i < 2800:
        #     continue


        # if 'Patrick sat gazing into space' in sentence:
        #     seen_last_halt_point = True            
        # if not seen_last_halt_point:
        #     continue
        test_tokens = cuda(reference_gpt2.to_tokens(sentence))
        # print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits, _ = demo_gpt2(test_tokens)        

        probabilities = torch.nn.functional.softmax(demo_logits[-1,-1], dim=0).detach().numpy()

        if probabilities[0] > 0.05:
            # Is or could be a complete sentence
            index_to_completeness[i] = 1
        else:
            index_to_completeness[i] = 0

        print(sentence)
        print(probabilities)
        print(probabilities.shape)
        print(probabilities[432])

        # for j in range(11):
        #     plt.plot(pickle.load(open(f'w_norm_sentence_completeness_layer_{j}.p', 'rb')).flatten())
        #     plt.show()

        test_tokens = cuda(reference_gpt2.to_tokens(sentence))
        # print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits, _ = demo_gpt2(test_tokens, intervene_in_resid_at_layer=3, 
                                   resid_intervention_filename='w_norm_sentence_completeness_layer_3.p',
                                   intervene_as_fraction_of_resid=0.05)

        probabilities = torch.nn.functional.softmax(demo_logits[-1,-1], dim=0).detach().numpy()

        if probabilities[0] > 0.05:
            # Is or could be a complete sentence
            index_to_completeness[i] = 1
        else:
            index_to_completeness[i] = 0

        print(sentence)
        print(probabilities)
        print(probabilities.shape)
        print(probabilities[432])        

        # add_streams_to_storage(resid_streams, all_resid_streams)

        # index_to_sentence[len(all_resid_streams[0]) - 1] = sentence

        if i % 50 == 0:
            print(i)
            # # pickle.dump(male_resid_streams, open(male_path, 'wb'))
            # # pickle.dump(female_resid_streams, open(female_path, 'wb'))
            # pickle.dump(all_resid_streams, open(all_path, 'wb'))
            # pickle.dump(index_to_completeness, open(index_to_completeness_path, 'wb'))
    # pickle.dump(male_resid_streams, open(male_path, 'wb'))
    # pickle.dump(female_resid_streams, open(female_path, 'wb'))
    # pickle.dump(all_resid_streams, open(all_path, 'wb'))
    # pickle.dump(index_to_completeness, open(index_to_completeness_path, 'wb'))





    raise Exception()


if False:
    import pickle
    from collections import defaultdict
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    male_path = 'male_resid_streams.p'
    female_path = 'female_resid_streams.p'
    all_path = 'resid_streams.p'
    male_resid_streams = pickle.load(open(male_path, 'rb')) if file_exists(male_path) else defaultdict(list)
    female_resid_streams = pickle.load(open(female_path, 'rb')) if file_exists(female_path) else defaultdict(list)
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    for i in range(11):
        male_resids = male_resid_streams[i]
        female_resids = female_resid_streams[i]

        male_vectors = [m.detach().numpy().flatten() for m in male_resids]
        female_vectors = [m.detach().numpy().flatten() for m in female_resids]

        print(len(male_vectors))
        print(len(female_vectors))

        X = male_vectors + female_vectors
        y = [0]*len(male_vectors) + [1]*len(female_vectors)
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model with training data
        clf = svm.SVC()
        clf.fit(X_train, y_train)

        # Predict on the test data
        y_pred = clf.predict(X_test)

        print("Classification report for classifier %s:\n%s\n"
            % (clf, classification_report(y_test, y_pred)))

    raise Exception()


if False:
    import pickle
    from sklearn.decomposition import PCA
    from collections import defaultdict

    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    from imblearn.over_sampling import SMOTE

    # Then use X_train_res and y_train_res to train your model


    all_path = 'eight_token_resid_streams.p'
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open('eight_token_index_to_sentence.p', 'rb'))
    index_to_completeness = pickle.load(open('eight_token_index_to_completeness.p', 'rb'))

    all_vectors_np_centered_list = []
    completeness_list = []

    for layer_index in range(11):
        print(f'Layer {layer_index}')

        all_resids = all_resid_streams[layer_index]        

        all_vectors = [m.detach().numpy().flatten() for m in all_resids]

        print(len(all_vectors))

        # Get vectors of a fixed length
        all_vectors_indices_fixed = [j for j, v in enumerate(all_vectors) if len(v) == 6912]
        all_vectors_fixed = [v for v in all_vectors if len(v) == 6912]

        print(len(all_vectors_fixed))

        print(len(index_to_completeness))
        print(len(index_to_sentence))

        all_vectors_np = np.array(all_vectors_fixed)

        # Center the data
        all_vectors_np_centered = all_vectors_np - np.mean(all_vectors_np, axis=0)
        all_vectors_np_centered_list.append(all_vectors_np_centered)
        
        # Add corresponding completeness values
        completeness = [index_to_completeness[all_vectors_indices_fixed[j]] for j in range(len(all_vectors_fixed))]
        completeness_list.extend(completeness)

        # random.shuffle(completeness)

        # Combine all the centered vectors
        # X = np.concatenate(all_vectors_np_centered_list, axis=0)
        X = all_vectors_np_centered

        # # Perform PCA on the data
        # pca = PCA(n_components=2000)
        # X_pca = pca.fit_transform(X)

        # Convert completeness list to a numpy array
        y = np.array(completeness)

        print(X.shape)
        print(y.shape)

        # print(X_pca.shape)

        print('hello')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # # Initialize the SMOTE object
        # sm = SMOTE(random_state=42)

        # Resample the training set
        # X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # Initialize the SVC object
        # clf = svm.LinearSVC(class_weight={0:1000, 1:1})
        # clf = svm.LinearSVC()
        clf = svm.SVC(max_iter=1000, kernel='linear', class_weight='balanced')

        # Train the model
        clf.fit(X_train, y_train)

        # Test the model
        y_pred = clf.predict(X_test)

        # Print the classification report and confusion matrix
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # Obtain the indices of the test set
        _, test_idx, _, _ = train_test_split(range(len(X)), y, test_size=0.2, random_state=42)

        # Predict labels for the test set
        y_pred = clf.predict(X_test)

        # Find false positives
        FP = np.nonzero((y_pred == 1) & (y_test == 0))[0]
        FP_original_indices = [test_idx[j] for j in FP]

        # Find false negatives
        FN = np.nonzero((y_pred == 0) & (y_test == 1))[0]
        FN_original_indices = [test_idx[j] for j in FN]

        print(f"Indices of false positives in original data: {FP_original_indices}")
        for j in FP_original_indices:
            print(index_to_sentence[all_vectors_indices_fixed[j]])

        print(f"Indices of false negatives in original data: {FN_original_indices}")

        for j in FN_original_indices:
            print(index_to_sentence[all_vectors_indices_fixed[j]])

        # Obtain the weights (coefficients) of the SVM model.
        w = clf.coef_[0]

        # Normalize the weight vector.
        w_norm = w / np.linalg.norm(w)

        w_norm_reshaped = np.reshape(w_norm, all_resid_streams[layer_index][0].shape)
        print(w_norm_reshaped)

        pickle.dump(w_norm_reshaped, open(f'w_norm_sentence_completeness_layer_{layer_index}.p', 'wb'))


    raise Exception()



if False:
    import pickle
    from sklearn.decomposition import PCA
    from collections import defaultdict

    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    from imblearn.over_sampling import SMOTE

    # Then use X_train_res and y_train_res to train your model


    all_path = 'eight_token_resid_streams.p'
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open('eight_token_index_to_sentence.p', 'rb'))
    index_to_completeness = pickle.load(open('eight_token_index_to_completeness.p', 'rb'))

    all_vectors_np_centered_list = []
    completeness_list = []

    for layer_index in range(11):
        print(f'Layer {layer_index}')

        all_resids = all_resid_streams[layer_index]        

        all_vectors = [m.detach().numpy().flatten() for m in all_resids]

        print(len(all_vectors))

        # Get vectors of a fixed length
        all_vectors_indices_fixed = [j for j, v in enumerate(all_vectors) if len(v) == 6912]
        all_vectors_fixed = [v for v in all_vectors if len(v) == 6912]

        print(len(all_vectors_fixed))

        print(len(index_to_completeness))
        print(len(index_to_sentence))

        all_vectors_np = np.array(all_vectors_fixed)

        # Center the data
        all_vectors_np_centered = all_vectors_np - np.mean(all_vectors_np, axis=0)
        all_vectors_np_centered_list.append(all_vectors_np_centered)
        
        # Add corresponding completeness values
        completeness = [index_to_completeness[all_vectors_indices_fixed[j]] for j in range(len(all_vectors_fixed))]
        completeness_list.extend(completeness)

        # random.shuffle(completeness)

        # Combine all the centered vectors
        # X = np.concatenate(all_vectors_np_centered_list, axis=0)
        X = all_vectors_np_centered

        # # Perform PCA on the data
        # pca = PCA(n_components=2000)
        # X_pca = pca.fit_transform(X)

        # Convert completeness list to a numpy array
        y = np.array(completeness)

        print(X.shape)
        print(y.shape)

        # print(X_pca.shape)

        print('hello')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # # Initialize the SMOTE object
        # sm = SMOTE(random_state=42)

        # Resample the training set
        # X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # Initialize the SVC object
        # clf = svm.LinearSVC(class_weight={0:1000, 1:1})
        # clf = svm.LinearSVC()
        clf = svm.SVC(max_iter=1000, kernel='linear', class_weight='balanced')

        # Train the model
        clf.fit(X_train, y_train)

        # Test the model
        y_pred = clf.predict(X_test)

        # Print the classification report and confusion matrix
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # Obtain the indices of the test set
        _, test_idx, _, _ = train_test_split(range(len(X)), y, test_size=0.2, random_state=42)

        # Predict labels for the test set
        y_pred = clf.predict(X_test)

        # Find false positives
        FP = np.nonzero((y_pred == 1) & (y_test == 0))[0]
        FP_original_indices = [test_idx[j] for j in FP]

        # Find false negatives
        FN = np.nonzero((y_pred == 0) & (y_test == 1))[0]
        FN_original_indices = [test_idx[j] for j in FN]

        print(f"Indices of false positives in original data: {FP_original_indices}")
        for j in FP_original_indices:
            print(index_to_sentence[all_vectors_indices_fixed[j]])

        print(f"Indices of false negatives in original data: {FN_original_indices}")

        for j in FN_original_indices:
            print(index_to_sentence[all_vectors_indices_fixed[j]])

        # Obtain the weights (coefficients) of the SVM model.
        w = clf.coef_[0]

        # Normalize the weight vector.
        w_norm = w / np.linalg.norm(w)

        w_norm_reshaped = np.reshape(w_norm, all_resid_streams[layer_index][0].shape)
        print(w_norm_reshaped)

        pickle.dump(w_norm_reshaped, open(f'w_norm_sentence_completeness_layer_{layer_index}.p', 'wb'))


    raise Exception()



if False:
    import pickle
    from sklearn.decomposition import PCA
    from collections import defaultdict

    all_path = 'eight_token_resid_streams.p'
    all_resid_streams = pickle.load(open(all_path, 'rb')) if file_exists(all_path) else defaultdict(list)

    index_to_sentence = pickle.load(open('eight_token_index_to_sentence.p', 'rb'))

    for i in range(11):
        print(f'Layer {i}')

        all_resids = all_resid_streams[i]        

        all_vectors = [m.detach().numpy().flatten() for m in all_resids]

        # print([len(k) for k in all_vectors])

        all_vectors_indices_fixed = [i for i, v in enumerate(all_vectors) if len(v) == 6912]
        all_vectors_np = np.array([v for v in all_vectors if len(v) == 6912])

        all_vectors_np_centered = all_vectors_np - np.mean(all_vectors_np, axis=0)

        pca = PCA(n_components=None)

        pca_all_vectors_centered = pca.fit_transform(all_vectors_np_centered)

        print(all_vectors_np_centered.shape)
        print(len(index_to_sentence))

        # print("Principal components:")
        # for i, component in enumerate(pca.components_[:100]):
        #     print(f"Component {i+1}: {component}")

        # Project the original vectors onto the principal components
        projections = np.dot(all_vectors_np_centered, pca.components_.T)


        # Sort the indices of the projections for each principal component
        sorted_indices = np.argsort(projections, axis=0)

        # Print the 10 vectors that are most and least in the direction of each principal component
        for i in range(min(10, pca.n_components_)):
            top_10 = sorted_indices[-10:, i]  # Indices of vectors most in direction
            bottom_10 = sorted_indices[:10, i]  # Indices of vectors least in direction
            
            print(f"Indices of vectors most in direction of component {i+1}: {top_10}")
            for j in top_10:
                print([index_to_sentence[all_vectors_indices_fixed[j]]])
            print('')
            # print(f"Vectors most in direction of component {i+1}: {[all_vectors[idx] for idx in top_10]}")

            print(f"Indices of vectors least in direction of component {i+1}: {bottom_10}")
            for j in bottom_10:
                print([index_to_sentence[all_vectors_indices_fixed[j]]])
            print('')            
            # print(f"Vectors least in direction of component {i+1}: {[all_vectors[idx] for idx in bottom_10]}")


    raise Exception()

if False:
    import pickle
    import IPython
    WEIRD_TOKENS = pickle.load(open('weird_tokens_int.p', 'rb'))
    # IPython.embed()
    def argmin_excluding_weird_tokens(tensor):
        exclude_indices = WEIRD_TOKENS

        # Create a mask for the elements to exclude
        mask = torch.ones_like(tensor, dtype=torch.bool)
        mask[exclude_indices] = False

        # Set the masked elements to a large value
        large_value = torch.finfo(tensor.dtype).max
        tensor_excluded = tensor.clone()
        tensor_excluded[mask == False] = large_value

        # Compute the argmin of the resulting tensor
        argmin_excluded = torch.argmin(tensor_excluded)

        return argmin_excluded


    def generate_random_string(length):
        # Define the set of characters to use (alphabetic characters and spaces)
        characters = string.ascii_letters + '         '

        # Generate a random string of the specified length
        random_string = ''.join(random.choice(characters) for _ in range(length))

        return random_string


    vocab_size = 50257
    running_probabilities = np.array([0.] * vocab_size)


    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"runnersllerhipISAnutsBILITYouls\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \" o2efnisdnp2e23-s\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"disclaimeruceStarsativelyitionallyports\". Reply: \""
    # test_string = f"Hello my name is Adam and I"
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"Barack Obama\". Reply: \""
    # test_string = f"Request: Please repeat the following string back to me: \"sasquatch\". Reply: \"sasquatch\". Request: Please repeat the following string back to me: \"abc def ghi\". Reply: \"abc def ghi\". Request: Please repeat the following string back to me: \"Alex Arkhipov\". Reply: \""
    # test_string = "When Tamarale and Tamastate went to the store, Tamarale gave a drink to"
    # test_string = "When Tamarale and Tamastate went to the store, Tamastate gave a drink to"
    # test_string = "When Tamastate and Tamarale went to the store, Tamarale gave a drink to"
    # test_string = "When Tamastate and Tamarale went to the store, Tamastate gave a drink to"
    # test_string = "When Guinevere and Romanova went to the store, Guinevere gave a drink to"
    # test_string = "When Romanova and Colberta went to the store, Romanova gave a drink to"
    # test_string = "When Tweedle Dee and Tweedle Dum went to the store, Tweedle Dum gave a drink to"
    # test_string = "When Alexandria-Solstice and Alexandria-Equinox went to the park, Alexandria-Solstice bought ice cream for Alex"
    # test_string = "1. e4 e5 2. Nf3 Nc6 3. Bc4"
    # test_string = " pourtant je ne sais pas pourquoi "
    # test_string_him = " He didn't understand, so I told"
    # test_string_her = " She didn't understand, so I told"
    # test_string_her = " Mary didn't understand, so I told"
    # test_string_her = " John didn't understand, so I told"
    # test_string_her = " The queen wanted more blankets, so I gave"
    # test_string = ' the president had a stroke. James'

    prompts = [
        "Despite the nationwide tension, the President held an aura of calmness that resonated with James",
        "In the silent White House corridor, the echoes of the President's words still lingered around James",
        "The President, true to his reputation, made a decision that left an indelible mark on James",
        "Once a doubter of the President's policies, a series of events triggered a change in James",
        "The invitation to the President's private event arrived unexpectedly and left a curious impression on James",
        "Over the years serving as a bodyguard, the President's true nature was slowly revealed to James",
        "The President's decision to pardon individuals raised questions about James",
        "The President's tweets often criticized James",        
        "Among the bustling crowd, the President's speech struck a deeply personal chord within James",
        "The secrets of the President, once unraveled, shifted the course of events for James",
        "Behind the grandeur of the President, there lay a trail of sacrifices known only to James",
        "The world watched as the President made history; a moment of profound significance for James"
    ]

    random_other_nouns = ['dog', 'cat', 'statue', 'robot', 'bug', 'boy', 'girl', 'man', 'woman', 'bird', 'orange', 'apple']

    # test_string_king = ' The king decreed that'
    # test_string_queen = ' The queen decreed that'

    # test_string_his = "He took something that wasn't"
    # test_string_hers = "She took something that wasn't"

    # resid = pickle.load(open('resid.p', 'rb'))
    # resid_zeros = torch.zeros(resid.shape)

    # print(resid_zeros[0][0][0])
    # resid_zeros[0][0][0] -= 1.

    # # visualize_tensor(resid_zeros, 'hello')

    # def get_output(mod_vector):
    #     demo_logits = demo_gpt2('abcd', load=True, load_with_mod_vector = mod_vector)

    #     return reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    def print_top_n_last_token_from_logits(my_logits,n, compare_on_these_token_indices):
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
        for i in topk_indices:
            if reference_gpt2.tokenizer.decode(i) in [' him', ' her', ' them']:
                print(f'index: {i} token: {reference_gpt2.tokenizer.decode(i)}')

        prob_token_list = list(zip(topk_probabilities, topk_tokens))
        prob_token_list.sort()
        # Print the top n tokens and their probabilities
        for probability, token in prob_token_list:
            print(f"Token: {token}, Probability: {probability}")
        if compare_on_these_token_indices:
            return [probabilities[index] for index in compare_on_these_token_indices]
        else:
            return None

    def layer_intervention_pairs_run_all(
            test_tokens_in,
            list_of_layer_intervention_pairs,
            compare_on_these_token_indices
    ):
        print("==== Default test ====")
        demo_logits_def = demo_gpt2(test_tokens_in, intervene_in_resid_at_layer=None,
                                resid_intervention_filename=None)
        default_token_probs = print_top_n_last_token_from_logits(demo_logits_def, 5, compare_on_these_token_indices)

        intervention_deltas = []
        for layer, intervention_file_name in list_of_layer_intervention_pairs:
            print(f"==== Intervention layer {layer} test file {intervention_file_name}====")
            demo_logits_int = demo_gpt2(test_tokens_in, intervene_in_resid_at_layer=layer,
                                    resid_intervention_filename=intervention_file_name,)
            new_token_probs = print_top_n_last_token_from_logits(demo_logits_int, 5,  compare_on_these_token_indices)

            intervention_deltas.append((layer, intervention_file_name, new_token_probs))

        for layer, intervention_file_name, new_token_probs in intervention_deltas:
            token_names = [reference_gpt2.tokenizer.decode(i) for i in compare_on_these_token_indices]
            print(f'Change For Intervention {intervention_file_name} on Layer {layer}:')
            for new_prob,old_prob, name in zip(new_token_probs, default_token_probs, token_names):
                print(f' Token {name}: {old_prob-new_prob} (from {old_prob} to {new_prob})')


    if False:
        print("=====Andrea Code ========")
        test_tokens_her = cuda(reference_gpt2.to_tokens(test_string_her))
        layer_intervention_pairs_run_all(
            test_tokens_her,
            [(i, f'andrea_small_update_{i}.p') for i in range(12)],
            [606, 607, 683],
        )

        raise Exception("Andrea Code Halt")

    import tiktoken
    import html
    import random
    from IPython.core.display import display, HTML
    def render_toks_w_weights(toks, weights):
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().detach()
        else:
            weights = torch.tensor(weights)
        if isinstance(toks, torch.Tensor):
            toks = decode(toks)
        highlighted_text = []
        print(weights)
        for weight, tok in zip(weights.tolist(), toks):
            if weight > 0.0:
                highlighted_text.append(f'<span style="background-color:rgba(135,206,250,{min(1.3*weight, 1)});border: 0.3px solid black;padding: 0.3px">{tok}</span>')
            else:
                highlighted_text.append(f'<span style="background-color:rgba(135,206,250,{min(-1.3*weight, 1)});border: 0.3px solid black;padding: 0.3px">{tok}</span>')
        highlighted_text = ''.join(highlighted_text)
        display(HTML(highlighted_text))
        print(highlighted_text)
        print(HTML(highlighted_text))
        return highlighted_text + "<br>"

    president_activations = torch.zeros((12, 3072))
    doctored_activations = torch.zeros((12, 3072))
    html_string = ''

    for test_string in prompts:
        for i in tqdm.tqdm(range(1)):


            doctored_test_string = test_string.replace('President', random.choice(random_other_nouns))
            print(test_string)

            # test_tokens_him = cuda(reference_gpt2.to_tokens(test_string_his))
            # print(reference_gpt2.to_str_tokens(test_tokens_him))

            # demo_logits = demo_gpt2(test_tokens_him, save_with_prefix='his')
            # zeros_arr_him_her[0][2][447] = -10.0
            # pickle.dump(zeros_arr_him_her, open('layer_2_position_447_neg_10.p', 'wb'))
            # pickle.dump(zeros_arr, open('zeros.p', 'wb'))

            import tiktoken
            test_tokens = cuda(reference_gpt2.to_tokens(test_string))
            doctored_test_tokens = cuda(reference_gpt2.to_tokens(doctored_test_string))

            enc = tiktoken.get_encoding('r50k_base')



            print(reference_gpt2.to_string(doctored_test_tokens))

            # print(test_tokens_her.shape)
            # test_tokens_king = cuda(reference_gpt2.to_tokens(test_string_king))
            # print(reference_gpt2.to_str_tokens(test_tokens_her))

            # demo_logits = demo_gpt2(test_tokens_her, intervene_in_resid_at_layer='start',
            #                         resid_intervention_filename='resid_him_minus_her_start.p')
            # demo_logits = demo_gpt2(test_tokens_her, intervene_in_resid_at_layer='start',
            #                         resid_intervention_filename='resid_him_minus_her_start_half.p')        
            # demo_logits = demo_gpt2(test_tokens_her, intervene_in_resid_at_layer=None,
            #                         resid_intervention_filename=None)
            # demo_logits = demo_gpt2(test_tokens_her, intervene_in_resid_at_layer=1,
            #                 resid_intervention_filename='layer_2_position_447_neg_10.p')
            demo_logits, demo_other_stuff = reference_gpt2.run_with_cache(test_tokens)
            doctored_logits, doctored_other_stuff = reference_gpt2.run_with_cache(doctored_test_tokens)

            COMEY_INDEX = 12388

            # print(demo_other_stuff)
            # print(demo_other_stuff['blocks.11.mlp.hook_post'].shape)

            # print([(k, enc.decode([j])) for k, j in enumerate(test_tokens)])

            # print(reference_gpt2.to_string(test_tokens))
            # print([demo_other_stuff[f'blocks.11.mlp.hook_post'][0,j,2498] for j in range(len(test_tokens))])
            # plt.plot([demo_other_stuff[f'blocks.11.mlp.hook_post'][0,j,2498] for j in range(len(test_tokens))])
            # plt.show()
            print(demo_other_stuff[f'blocks.11.mlp.hook_post'].shape)
            print(demo_other_stuff[f'blocks.11.mlp.hook_post'][0,:,2498])
            print([(token, weight) for (token, weight) in zip(reference_gpt2.to_str_tokens(test_string), 
                                                              demo_other_stuff[f'blocks.11.mlp.hook_post'][0,:,2498])])

            html_string += str(render_toks_w_weights(reference_gpt2.to_str_tokens(test_string), demo_other_stuff[f'blocks.11.mlp.hook_post'][0,:,2498]))
            html_string += str(render_toks_w_weights(reference_gpt2.to_str_tokens(doctored_test_string), doctored_other_stuff[f'blocks.11.mlp.hook_post'][0,:,2498]))

            for j in range(12):
                president_activations[j] += demo_other_stuff[f'blocks.{j}.mlp.hook_post'][0,-1,:]
                doctored_activations[j] += doctored_other_stuff[f'blocks.{j}.mlp.hook_post'][0,-1,:]

            # print(demo_logits.shape)

            demo_comey_logit = demo_logits[0,-1,COMEY_INDEX]
            doctored_comey_logit = doctored_logits[0,-1,COMEY_INDEX]

            # print(demo_comey_logit - doctored_comey_logit)

            # Get the logits for the last predicted token
            last_logits = demo_logits[-1, -1]
            doctored_last_logits = doctored_logits[-1, -1]


            # print(last_logits[COMEY_INDEX] - doctored_last_logits[COMEY_INDEX])
            # Apply softmax to convert the logits to probabilities
            probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
            doctored_probabilities = torch.nn.functional.softmax(doctored_last_logits, dim=0).detach().numpy()

            print(probabilities[COMEY_INDEX], doctored_probabilities[COMEY_INDEX])

            # Get the indices of the top 10 probabilities
            topk_indices = np.argpartition(probabilities, -10)[-10:]
            # Get the top 10 probabilities
            topk_probabilities = probabilities[topk_indices]
            # Get the top 10 tokens
            topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

            # Print the top 10 tokens and their probabilities
            # for token, probability in zip(topk_tokens, topk_probabilities):
            #     print(f"Token: {token}, Probability: {probability}")

            # Get the most probable next token and append it to the test string
            most_probable_next_token = reference_gpt2.tokenizer.decode(last_logits.argmax())
            # print(probabilities[683])
            # print(probabilities[12388])
            # for token, probability in zip(topk_tokens, topk_probabilities):
            #     print(f"Token: {token}, Probability: {probability}")


                # raise Exception()

            test_string += most_probable_next_token

            # print(test_string)
            # if i == 0:
            #     test_string += ' Tam'
            # else:
            #     test_string += most_probable_next_token

    print(html_string)

    diff_activations = president_activations - doctored_activations
    flattened_activations = diff_activations.flatten()

    for j in range(12):
        plt.plot(diff_activations[j], label=f"Activation {j}")

    plt.legend()
    plt.show()

    # Getting the top 10 activations overall
    top_activations_indices = np.argpartition(flattened_activations, -10)[-10:]
    top_activations_values = flattened_activations[top_activations_indices]

    # Printing the top 10 activations
    for i in range(10):
        layer_index, within_layer_index = np.unravel_index(top_activations_indices[i], diff_activations.shape)
        print(f"Top activation {i+1} is in layer {layer_index+1} at index {within_layer_index} with a value of {top_activations_values[i]}")
        
    # for i in tqdm.tqdm(range(5)):
    #     test_tokens = cuda(reference_gpt2.to_tokens(test_string))
    #     print(reference_gpt2.to_str_tokens(test_tokens))

    #     demo_logits = demo_gpt2(test_tokens)#, display=i==0, load=i==0, load_with_mod_vector = None)
    #     print(demo_logits[-1, -1].shape)
        
    #     logits = demo_logits[-1, -1].detach().numpy()
    #     # running_probabilities = np.multiply(running_probabilities, logits)
    #     # running_probabilities = logits
    #     # adjusted_probabilities = logits - running_probabilities
    #     # running_probabilities = running_probabilities * i / (i + 1) + logits / (i + 1)
    #     # running_probabilities = logits

    #     # test_string += reference_gpt2.tokenizer.decode(argmin_excluding_weird_tokens(torch.from_numpy(adjusted_probabilities)))

    #     # print(demo_logits[-1, -1][demo_logits[-1, -1].argmax()])



    #     test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    #     # test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmin())
    #     # test_string += reference_gpt2.tokenizer.decode(argmin_excluding_weird_tokens(demo_logits[-1, -1]))

    # print(test_string)
    raise Exception()
    


if False:

    # test_string = f"a\nable\nabout\nabove\naccident\nacid\nacross\n"
    # test_string = f' 323 + 581 = 904; 241 + 521 = 762; 421 + 512 ='
    # test_string = f' 323 + 581 = 904; 241 + 521 = 762; 323 + 2018 ='
    test_string = f' 323 is greater than 153. 457 is less than 561. 552362 is less than'
    # test_string = "I don't know if"
    for i in tqdm.tqdm(range(5)):
        test_tokens = cuda(reference_gpt2.to_tokens(test_string))
        print(reference_gpt2.to_str_tokens(test_tokens))
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    print(test_string)
    print(get_pretrained_model_config(model_name))

    raise Exception()


if False:
    model_centered = HookedTransformer.from_pretrained("gpt2-small")

    model_not_centered = HookedTransformer.from_pretrained("gpt2-small", center_writing_weights=False)

    mat = model_not_centered.W_pos.detach().numpy()

    from math import log10

    # Calculate SVD
    U, S, VT = np.linalg.svd(mat[20:])

    print(mat.shape)

    print(U.shape)
    print(VT.shape)

    # Threshold for singular values
    threshold = 0.01 * np.max(S)

    # Plot singular values in log scale
    plt.figure(figsize=(10, 6))
    plt.plot([log10(x) if x > 0 else 0 for x in S])

    # Draw horizontal line at threshold
    plt.axhline(y=log10(threshold), color='r', linestyle='dotted')

    # Draw vertical line at the point where singular values drop below threshold
    below_threshold_indices = np.where(S < threshold)[0]
    if below_threshold_indices.size > 0:
        plt.axvline(x=below_threshold_indices[0], color='r', linestyle='dotted')

    # Set axis labels
    plt.xlabel('Index')
    plt.ylabel('Log10(Singular Value)')

    plt.title('Singular Values in Log Scale')

    plt.show()

    # Plot the first 10 left singular vectors
    plt.figure(figsize=(10, 6))

    for i, vec in enumerate(U[:10]):
        plt.plot(vec, label='Vector {}'.format(i+1))

    # Adding legend
    plt.legend(loc='best')

    # Adding grid
    plt.grid(True)

    # Adding title and labels
    plt.title('First 10 Left Singular Vectors')
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


    # plt.matshow(model_not_centered.W_pos.detach().numpy())
    # plt.show()

    raise Exception()



# raise Exception()
# Choose the number of clusters (k)
k = 200

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import IPython
from collections import defaultdict
import tiktoken

IPython.embed()

matrix = demo_gpt2.embed.W_E.detach().numpy()

# enc = tiktoken.encode('he')

# for row in matrix[:30]:
plt.plot(matrix[0])
plt.plot(matrix[43453])


# plt.matshow(matrix)
plt.show()

def to_single_str_token(k: int) -> str:
    return reference_gpt2.to_str_tokens(torch.tensor([k, 1]))[0]

# IPython.embed()
#
print(matrix.shape)

def can_be_inted(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Keep only tokens that are numbers with a leading space

# new_matrix = []
# target_array = []

train_X = []
test_X = []
train_Y = []
test_Y = []

# random.shuffle(matrix)

# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)
#     if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and int(str_token) < 700:
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and not (int(str_token) > 1600 and int(str_token) < 2100):
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 1600 and int(str_token) < 2100:
#         print(f'Keeping token {i}: "{str_token}"')

#         if random.random() < 0.1:
#             test_X.append(matrix[i])
#             test_Y.append(int(str_token))
#         else:
#             train_X.append(matrix[i])
#             train_Y.append(int(str_token))

#         # new_matrix.append(matrix[i])
#         # target_array.append(int(str_token))
#         # target_array.append(int(i))

def is_numeric_string(s):
    for c in s:
        if c not in '0123456789':
            return False
    return True

# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and int(str_token) < 700:
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and not (int(str_token) > 1600 and int(str_token) < 2100):
#     if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token[1:]) < 2100:
#         print(f'Keeping token {i}: "{str_token}"')

#         if int(str_token[1:]) < 1600:
#         # if int(str_token[1:]) > 1600 and int(str_token[1:]) < 2100:
#             if random.random() < 0.1:
#                 test_X.append(matrix[i])
#                 test_Y.append(int(str_token))
#             else:
#                 train_X.append(matrix[i])
#                 train_Y.append(int(str_token))

#         elif int(str_token[1:]) < 2100:
#         # else:
#             test_X.append(matrix[i])
#             test_Y.append(int(str_token))

#     else:
#         if random.random() < 0.01:
#             test_X.append(matrix[i])
#             test_Y.append(0)

        # new_matrix.append(matrix[i])
        # target_array.append(int(str_token))
        # target_array.append(int(i))

# from string import ascii_letters

# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)    
#     if len(str_token) > 1 and str_token[0] == ' ' and str_token[1] in ascii_letters:
#         if random.random() < 0.1:
#             test_X.append(matrix[i])
#             test_Y.append(string.ascii_letters.index(str_token[1]))

#         else:
#             train_X.append(matrix[i])
#             train_Y.append(string.ascii_letters.index(str_token[1]))

from string import ascii_lowercase

char_index = 5

for i in range(matrix.shape[0]):
    str_token = to_single_str_token(i)    
    if len(str_token) > char_index and str_token[0] == ' ' and str_token[char_index] in ascii_lowercase:
        if random.random() < 0.2:
            test_X.append(matrix[i])
            test_Y.append(string.ascii_lowercase.index(str_token[char_index]))

        else:
            train_X.append(matrix[i])
            train_Y.append(string.ascii_lowercase.index(str_token[char_index]))


# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and int(str_token) < 700:
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and not (int(str_token) > 1600 and int(str_token) < 2100):
#     # if is_numeric_string(str_token) and len(str_token) <= 10:
#     if str_token[0] == ' ' and can_be_inted(str_token[1:]) and len(str_token) <= 10:
#         print(f'Keeping token {i}: "{str_token}"')

#         if random.random() < 0.1:
#             test_X.append(matrix[i])
#             test_Y.append(len(str_token))
#         else:
#             train_X.append(matrix[i])
#             train_Y.append(len(str_token))

#         # new_matrix.append(matrix[i])
#         # target_array.append(int(str_token))
#         # target_array.append(int(i))


# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and int(str_token) < 700:
#     # if str_token[0] == ' ' and can_be_inted(str_token[1:]) and int(str_token) > 0 and not (int(str_token) > 1600 and int(str_token) < 2100):
#     if random.random() < 0.02 and len(str_token) <= 10:
#         print(f'Keeping token {i}: "{str_token}"')

#         if random.random() < 0.1:
#             test_X.append(matrix[i])
#             test_Y.append(len(str_token))
#         else:
#             train_X.append(matrix[i])
#             train_Y.append(len(str_token))

#         # new_matrix.append(matrix[i])
#         # target_array.append(int(str_token))
#         # target_array.append(int(i))


def is_alphabetic_string(s):
    for c in s:
        if not c.isalpha():
            return False
    return True



# for i in range(matrix.shape[0]):
#     str_token = to_single_str_token(i)
#     if str_token[0] == ' ' and is_alphabetic_string(str_token[1:]) and alphabetic_rank(str_token[1:]) <= 27:
#         if random.random() < 0.1:
#             test_X.append(matrix[i])
#             test_Y.append(alphabetic_rank(str_token[1:]))
#         else:
#             train_X.append(matrix[i])
#             train_Y.append(alphabetic_rank(str_token[1:]))
#         # if alphabetic_rank(str_token[1:]) > 30:
#         #     print(alphabetic_rank(str_token[1:]))
#         #     print(str_token[1:])
#         # # print(alphabetic_rank(str_token[1:]))


train_X = np.array(train_X)
test_X = np.array(test_X)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)
# print(new_matrix.shape)

print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np


# def ridge_regression(X, Y, alpha=1.0):
#     # Convert X and Y to numpy arrays
#     X = np.array(X)
#     Y = np.array(Y)

#     # Create a Ridge Regression object
#     regr = Ridge(alpha=alpha)

#     # Fit the model to our data
#     regr.fit(X, Y)

#     # Print the coefficients of the linear regression model
#     print('Coefficients: ', regr.coef_)

#     # Print the intercept of the model
#     print('Intercept: ', regr.intercept_)
    
#     # Predict values using the model
#     Y_pred = regr.predict(X)
    
#     # Plot actual vs predicted values
#     plt.scatter(Y, Y_pred, color='blue')
    
#     # Plot y=x line. Perfect predictions would fall on this line.
#     plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linewidth=2)
    
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.title('Actual vs Predicted')
    
#     plt.show()
    
#     return regr


import seaborn as sns


def logistic_regression(X_train, Y_train, X_test, Y_test):
    # Assuming that you have a DataFrame 'df' and you are going to predict 'target_column'
    # Replace 'feature1', 'feature2',...,'featureN' with your real feature column names

    # Instantiating the model (using the default parameters)
    logreg = LogisticRegression()

    # Fitting the model with data
    logreg.fit(X_train, Y_train)

    # Predicting the response for the test dataset
    y_pred = logreg.predict(X_test)

    # Checking accuracy
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

    # You can also get the confusion matrix
    print("Confusion Matrix:\n", metrics.confusion_matrix(Y_test, y_pred))

    # And the classification report
    print("Classification Report:\n", metrics.classification_report(Y_test, y_pred))    

    # Get the unique class labels
    class_labels = np.unique(np.concatenate((Y_test, y_pred)))

    # Create the confusion matrix
    confusion_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for i in range(len(Y_test)):
        true_label_index = np.where(class_labels == Y_test[i])[0][0]
        pred_label_index = np.where(class_labels == y_pred[i])[0][0]
        confusion_matrix[true_label_index, pred_label_index] += 1

    # Create a heatmap for the confusion matrix
    fig, ax = plt.subplots()
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(string.ascii_lowercase[:len(class_labels)]), yticklabels=list(string.ascii_lowercase[:len(class_labels)]), ax=ax)


    # Add labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    # Display the plot
    plt.show()


def ridge_regression(X_train, Y_train, X_test, Y_test, alpha=1.0):
    # Convert to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Create a Ridge Regression object
    regr = Ridge(alpha=alpha)

    # Fit the model to the training data
    regr.fit(X_train, Y_train)

    # Print the coefficients of the linear regression model
    print('Coefficients: ', regr.coef_)

    # Print the intercept of the model
    print('Intercept: ', regr.intercept_)
    
    # Predict values using the model for both training and test data
    Y_train_pred = regr.predict(X_train)
    Y_test_pred = regr.predict(X_test)
    
    # Calculate and print the Mean Squared Error for training and test data
    print('Training Mean Squared Error: ', mean_squared_error(Y_train, Y_train_pred))
    print('Test Mean Squared Error: ', mean_squared_error(Y_test, Y_test_pred))
    
    # Plot actual vs predicted values for training data in blue
    plt.scatter(Y_train, Y_train_pred, color='blue', label='Training data')
    
    # Plot actual vs predicted values for test data in red
    plt.scatter(Y_test, Y_test_pred, color='red', label='Test data')
    
    # Plot y=x line. Perfect predictions would fall on this line.
    plt.plot([min(Y_train.min(), Y_test.min()), max(Y_train.max(), Y_test.max())], 
             [min(Y_train.min(), Y_test.min()), max(Y_train.max(), Y_test.max())], color='black', linewidth=2)
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Regression on alphabetic ranks of alphabetical tokens (with leading space)')
    plt.legend()
    
    plt.show()
    
    return regr

def loo_cv(X, Y, alpha=1.0):
    # Convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Create a Ridge Regression object
    regr = Ridge(alpha=alpha)
    
    # Create a LeaveOneOut object
    loo = LeaveOneOut()
    
    errors = []

    # Perform Leave-One-Out cross-validation
    for train_index, test_index in loo.split(X):
        print(test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # Fit the model to the training data
        regr.fit(X_train, Y_train)
        
        # Predict the target for the test data
        Y_pred = regr.predict(X_test)
        
        # Calculate the error
        error = mean_squared_error(Y_test, Y_pred)
        
        errors.append(error)

        if test_index[0] > 50:
            break
    
    # Plot the errors
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Observation')
    plt.ylabel('Error')
    plt.title('Leave-One-Out Cross-Validation Errors')
    plt.show()

    return errors

# Example usage:
# model = ridge_regression(train_X, train_Y, test_X, test_Y, alpha=1.0)
model = logistic_regression(train_X, train_Y, test_X, test_Y)
# model = loo_cv(new_matrix, target_array)


# Example usage:
# X = [[0, 0], [1, 1], [2, 2]]
# Y = [0, 1, 2]
# model = linear_regression(X, Y)

# plt.plot()

raise Exception()


# Perform k-means clustering on the rows of the matrix
kmeans = KMeans(n_clusters=k, random_state=0).fit(matrix)

clusters = defaultdict(list)
clusters_int = defaultdict(list)
for i, label in enumerate(kmeans.labels_):
    clusters[label].append(to_single_str_token(i))
    clusters_int[label].append(i)

# Print cluster assignments for each row
print("Cluster assignments:", kmeans.labels_)

# Print the centroids (mean of each cluster)
print("Centroids:", kmeans.cluster_centers_)

IPython.embed()

# demo_gpt2.embed.W_E


print(test_string)