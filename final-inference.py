#!/usr/bin/env python3

import argparse
import math
import json
import mmap
import pathlib
import sys

import regex
import torch


class Tokenizer:

    def __init__(self, tokenizer_path: str):
        with open(tokenizer_path) as f:
            tokenizer_data = json.load(f)
        split = next(filter(lambda t: t["type"] == "Split", tokenizer_data["pre_tokenizer"]["pretokenizers"]))
        self.split_regex = regex.compile(split["pattern"]["Regex"])

        # space is encoded as Ġ, for simplicity, just use space here.
        self.vocab = {k.replace("Ġ", " ").replace("Ċ", "\n").encode("utf-8"): v for k, v in tokenizer_data["model"]["vocab"].items()}
        added_tokens = {t["content"]: t["id"] for t in tokenizer_data["added_tokens"]}

        self.begin_of_text = added_tokens["<|begin_of_text|>"]
        self.end_of_text = added_tokens["<|end_of_text|>"]

        self.vocab.update(added_tokens)

        # inverse vocabulary for detokenization
        self.vocab_inv = { v: k for k, v in self.vocab.items() }

    def tokenize(self, text: str) -> list[int]:
        str_tokens = self.split_regex.findall(text)

        # Add specific markers for beginning and end of text
        # str_tokens = ["<|begin_of_text|>"] + str_tokens + ["<|end_of_text|>"]

        tokens = []

        for str_token in str_tokens:
            parts = [bytes([b]) for b in str_token.encode("utf-8")]

            while True:
                # Iterate over all pairs and find the pair we want to merge the most
                min_idx = None
                min_rank = None
                for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                    rank = self.vocab.get(pair[0] + pair[1])
                    if rank is not None and (min_rank is None or rank < min_rank):
                        min_idx = i
                        min_rank = rank

                # If there were no pairs we could merge, we're done!
                if min_rank is None:
                    break
                assert min_idx is not None

                # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
                parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

            tokens.extend(self.vocab[part] for part in parts)

        return [self.begin_of_text] + tokens

    def detokenize(self, tokens: list[int]) -> str:
        decoded = b""
        for t in tokens:
            decoded += self.vocab_inv[t]

        return decoded.decode("utf-8")


def load_raw_model(path: str, device):
    model_dir = pathlib.Path(path)
    model = {}
    with open(model_dir / "metadata.json") as file:
        metadata = json.load(file)
    for tensor_name, tensor_metadata in metadata.items():
        if tensor_name == "__metadata__":
            continue
        file_path = model_dir / f"{tensor_name}.raw"
        tensor_shape = tensor_metadata["shape"]
        size = torch.prod(torch.tensor(tensor_shape))
        model[tensor_name] = torch.from_file(str(file_path), dtype=torch.float32, size=size).to(device).reshape(tensor_shape)
    return model

### Inference ######################################################################################

def silu(x):
    """
    Sigmoid Linear Unit
    The SiLU function is also known as the swish function.
    """
    return x * (1 / (1 + torch.exp(-x)))


from typing import Any


class AttentionBlock:

    def __init__(self, layer_index: int, config: dict[str, Any], weights: dict[str, torch.Tensor], freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_attention_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.max_seq_len = config["max_seq_len"]

        self.q_weight = weights[f"model.layers.{layer_index}.self_attn.q_proj.weight"].T
        self.k_weight = weights[f"model.layers.{layer_index}.self_attn.k_proj.weight"].T
        self.v_weight = weights[f"model.layers.{layer_index}.self_attn.v_proj.weight"].T
        self.o_weight = weights[f"model.layers.{layer_index}.self_attn.o_proj.weight"].T

        self.k_cache = torch.zeros((self.max_seq_len, self.num_key_value_heads, self.head_dim))
        self.v_cache = torch.zeros((self.max_seq_len, self.num_key_value_heads, self.head_dim))


    def apply_rotary_positional_encoding(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int) -> torch.Tensor:
        input_length = xq.shape[0]

        freqs_cos = self.freqs_cos[start_pos: start_pos + input_length]
        freqs_sin = self.freqs_sin[start_pos: start_pos + input_length]
        # split the last dimension of the embedding in two channels to produce real, imaginary pairs.

        ### TODO: this isn't the same operation as rotate_half of the tf version
        ### r and i parts are contiguous, in tf, they are separated by x.shape[-1] // 2

        # Split xq and xk as a complex number representation
        xq_r, xq_i = torch.chunk(xq, 2, dim=-1)
        xk_r, xk_i = torch.chunk(xk, 2, dim=-1)

        freqs_cos = torch.unsqueeze(freqs_cos, dim=1)
        freqs_sin = torch.unsqueeze(freqs_sin, dim=1)

        # Apply rotation using real numbers.
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        xq_out = torch.cat([xq_out_r, xq_out_i], dim=-1)
        xk_out = torch.cat([xk_out_r, xk_out_i], dim=-1)

        return xq_out, xk_out


    def __call__(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor | None) -> torch.Tensor:
        # This code isn't batched, so `x` contains only a single token

        input_length = x.shape[0]

        # Compute query, key and value vectors for this token

        xq = x @ self.q_weight
        xk = x @ self.k_weight
        xv = x @ self.v_weight

        xq = xq.reshape((input_length, self.num_attention_heads, self.head_dim))
        xk = xk.reshape((input_length, self.num_key_value_heads, self.head_dim))
        xv = xv.reshape((input_length, self.num_key_value_heads, self.head_dim))
        xq, xk = self.apply_rotary_positional_encoding(xq, xk, start_pos)

        # Populate KV cache
        self.k_cache[start_pos: start_pos + input_length] = xk
        self.v_cache[start_pos: start_pos + input_length] = xv

        # Extract all key and values up to the current one from the cache.
        ks = self.k_cache[: start_pos + input_length]
        vs = self.v_cache[: start_pos + input_length]

        repeats = self.num_attention_heads // self.num_key_value_heads
        xk = torch.repeat_interleave(ks, repeats, dim=1)
        xv = torch.repeat_interleave(vs, repeats, dim=1)

        # ["L, HN, HD"] -> ["HN, L, HD"]
        xq = xq.transpose(0, 1)
        xk = xk.transpose(0, 1)
        xv = xv.transpose(0, 1)

        # flip the last dimensions to allow matrix multiplication
        xk = xk.transpose(2, 1)
        attention = xq @ xk
        attention = attention / self.sqrt_head_dim

        # Mask is only used at the beginning when processing the input tokens
        if mask is not None:
            attention = attention + mask[None, :, :]
        attention = attention.softmax(dim=-1)

        output = attention @ xv

        # ["HN, L or 1, HD"] -> ["L or 1, D"]
        output = output.transpose(0, 1).reshape(input_length, -1)
        output = output @ self.o_weight

        return output


class FeedForward:
    def __init__(self, layer_index: int, weights: dict[str, torch.Tensor]):
        self.up_weight = weights[f"model.layers.{layer_index}.mlp.up_proj.weight"].T
        self.down_weight = weights[f"model.layers.{layer_index}.mlp.down_proj.weight"].T
        self.gate_weight = weights[f"model.layers.{layer_index}.mlp.gate_proj.weight"].T

    def __call__(self, x: torch.Tensor):
        swish = silu(x @ self.gate_weight)
        x_V = x @ self.up_weight
        x = swish * x_V
        x = x @ self.down_weight
        return x


class RMSNorm:

    def __init__(self, weights_array: torch.Tensor, eps: float):
        self.weights = weights_array
        self.eps = eps

    def __call__(self, x: torch.Tensor):
        x_squared = x ** 2
        rms = torch.sqrt(x_squared.mean(-1, keepdims=True) + self.eps)
        rms_norm = (x / rms) * self.weights
        return rms_norm


class TransformerBlock:
    def __init__(self, layer_index: int, config: dict[str, Any], weights: dict[str, torch.Tensor], freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        self.attention = AttentionBlock(layer_index, config, weights, freqs_cos, freqs_sin)

        self.feed_forward = FeedForward(layer_index, weights)

        self.input_layernorm = RMSNorm(
            weights.get(f"model.layers.{layer_index}.input_layernorm.weight"),
            eps=config["rms_norm_eps"]
        )
        self.post_attention_layernorm = RMSNorm(
            weights.get(f"model.layers.{layer_index}.post_attention_layernorm.weight"),
            eps=config["rms_norm_eps"]
        )

    def __call__(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor):
        # RMSNorm
        norm_x = self.input_layernorm(x)

        # Masked Multi-Head Attention
        h1 = self.attention(norm_x, start_pos, mask)

        z = x + h1

        # RMSNorm
        norm_z = self.post_attention_layernorm(z)
        # Feed Forward + SwiGLU
        h2 = self.feed_forward(norm_z)
        out = z + h2

        return out



class Llama:
    def __init__(self, model_path: str, config_path: str, max_seq_len: int, device):
        weights = load_raw_model(model_path, device)

        with open(config_path) as file:
            config = json.load(file)

        self.config = config

        config["max_seq_len"] = max_seq_len

        self.token_embeddings = weights.get("model.embed_tokens.weight")

        # RoPE #1
        # freqs_cos, freqs_sin = self.compute_cos_sin_cache(
        #     config["hidden_size"] // config["num_attention_heads"],
        #     config["max_seq_len"],
        #     config["rope_theta"],
        # )
        freqs_cos, freqs_sin = self.precompute_rope_factors()

        self.layers = []
        for layer_index in range(config["num_hidden_layers"]):
            self.layers.append(TransformerBlock(layer_index, config, weights, freqs_cos, freqs_sin))

        self.norm = RMSNorm(weights.get("model.norm.weight"), eps=config["rms_norm_eps"])
        # self.lm_head_weight = weights.get("lm_head.weight").T
        self.lm_head_weight = weights.get("model.embed_tokens.weight").T


    def compute_cos_sin_cache(self, head_dim: int, max_seq_len: int, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)

        return torch.cos(freqs), torch.sin(freqs)


    def compute_llama3_inv_freq(self):
        config = self.config

        head_dim = config["head_dim"]
        base = config["rope_theta"]

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )

        rope_scaling = config.get("rope_scaling", None)
        if rope_scaling is None or rope_scaling.get("rope_type") != "llama3":
            return inv_freq

        factor = float(rope_scaling["factor"])
        low_freq_factor = float(rope_scaling["low_freq_factor"])
        high_freq_factor = float(rope_scaling["high_freq_factor"])
        old_context_len = int(rope_scaling["original_max_position_embeddings"])

        wavelen = 2 * math.pi / inv_freq
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,
            inv_freq,
        )

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        inv_freq_llama = torch.where(
            is_medium_freq,
            smoothed_inv_freq,
            inv_freq_llama,
        )

        return inv_freq_llama

    def precompute_rope_factors(self):
        inv_freq = self.compute_llama3_inv_freq()
        positions = torch.arange(self.config["max_seq_len"], dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        # freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        freqs_cos = torch.cos(freqs).to(torch.float32)
        freqs_sin = torch.sin(freqs).to(torch.float32)
        return freqs_cos, freqs_sin

    def __call__(self, input_ids, start_pos: int):
        input_length = input_ids.shape[0]
        h = self.token_embeddings[input_ids]

        # `mask` is generated only once at the beginning.
        mask = None
        if input_length > 1:
            mask = torch.full((input_length, input_length), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.cat([torch.zeros((input_length, start_pos)), mask], axis=1)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, mask)

        # RMSNorm
        h = self.norm(h)
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        logit = h[[-1], :] @ self.lm_head_weight
        return logit

    def sample(self, logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
        logits = logits.squeeze()

        if temperature == 0:
            return torch.argmax(logits)

        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        return next_id[0]

    def generate(self, input_ids, temperature, max_new_tokens: int):
        input_length = input_ids.shape[0]
        next_id = None

        for i, curr_pos in enumerate(range(input_length, input_length + max_new_tokens)):
            if i == 0:  # Prefill phase
                inputs = input_ids
                pos = 0
            else:  # Decode phase
                inputs = next_id.unsqueeze(0)
                pos = curr_pos

            logits = self(inputs, pos)
            next_id = self.sample(logits, temperature)
            yield next_id


def inference(prompt: str, device_name: str, temperature: float, max_new_tokens: int):
    device = torch.device(device_name) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(device)
    tokenizer = Tokenizer("models/Llama-3.2-1B/tokenizer.json")

    input_ids = torch.tensor([tokenizer.tokenize(prompt)])[0]

    llama = Llama("models/Llama-3.2-1B/tensors-fp32", "models/Llama-3.2-1B/config.json", max_new_tokens + len(input_ids), device)

    print(prompt, end="")
    for id in llama.generate(input_ids, temperature, max_new_tokens=max_new_tokens):
        id = int(id.item())
        if id == tokenizer.end_of_text:
            break
        print(tokenizer.detokenize([id]), end="")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Model temperature")
    parser.add_argument("--max-new-tokens", "-m", type=int, default=300, help="Max new tokens")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="CUDA device to use")
    parser.add_argument("prompt", nargs="*", default=(), help="Prompt")
    args = parser.parse_args()

    inference(" ".join(args.prompt), args.device, args.temperature, args.max_new_tokens)