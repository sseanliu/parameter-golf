"""
Parameter Golf Autoresearch -- Training Script.
This is the ONLY file the agent modifies.
Usage: python train.py > run.log 2>&1
"""

import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prepare import (
    TIME_BUDGET, VOCAB_SIZE, DEFAULT_SEQ_LEN,
    TokenStream, load_validation_tokens,
    evaluate_model, evaluate_sliding_window,
    save_and_reload_quantized, compute_artifact_size,
)

# ===========================================================================
# HYPERPARAMETERS (tune these!)
# ===========================================================================

NUM_LAYERS = 10
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 2
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
LOGIT_SOFTCAP = 30.0
TRAIN_SEQ_LEN = DEFAULT_SEQ_LEN
TRAIN_BATCH_TOKENS = 524_288
TIED_EMBED_INIT_STD = 0.005

# Optimizer
TIED_EMBED_LR = 0.10
MATRIX_LR = 0.04
SCALAR_LR = 0.04
MUON_MOMENTUM = 0.95
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.85
MUON_MOMENTUM_WARMUP_STEPS = 500
WARMDOWN_ITERS = 2500
WARMUP_STEPS = 20
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
MUON_WEIGHT_DECAY = 0.02

# Eval
EVAL_STRIDE = 64  # sliding window stride (0 = non-overlapping)
EVAL_BATCH_SEQS = 256

# ===========================================================================
# CONTROL TENSOR PATTERNS (kept in fp32/fp16 during quantization)
# ===========================================================================

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

# ===========================================================================
# MUON OPTIMIZER
# ===========================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


# ===========================================================================
# TRANSFORMER MODEL
# ===========================================================================

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._cache = (0, None, None)

    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len or self._cache[1] is None or self._cache[1].device != device:
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                base = self.base * (scale ** (self.dim / (self.dim - 2)))
                inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            else:
                inv = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv.dtype)
            freqs = torch.outer(t, inv)
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len=1024):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.num_encoder_layers = NUM_LAYERS // 2
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers
        self.num_skip = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, MODEL_DIM, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT, TRAIN_SEQ_LEN)
            for _ in range(NUM_LAYERS)
        ])
        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=TIED_EMBED_INIT_STD)
        # Overtone spectral init
        with torch.no_grad():
            U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
            target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
            self.tok_emb.weight.data = (U * target_S[None, :]) @ V
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)
        # Phase-transition residual mix
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(NUM_LAYERS - 1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(MODEL_DIM)
                block.resid_mix.data[1] = (1 - phase) * torch.ones(MODEL_DIM)

    def forward(self, input_ids, target_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (MODEL_DIM,))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x.reshape(-1, MODEL_DIM), self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (MODEL_DIM,))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight.to(x.dtype))
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def main():
    global zeropower_via_newtonschulz5

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Data
    val_tokens = load_validation_tokens(TRAIN_SEQ_LEN)
    stream = TokenStream()
    grad_accum_steps = 8

    # Model
    model = GPT().to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    # Restore small params to fp32
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_params: {n_params}")

    # Optimizers
    block_params = list(model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_params if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    opt_tok = torch.optim.AdamW([{"params": [model.tok_emb.weight], "lr": TIED_EMBED_LR, "base_lr": TIED_EMBED_LR}],
                                 betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY, fused=True)
    opt_muon = Muon(matrix_params, lr=MATRIX_LR, momentum=MUON_MOMENTUM, backend_steps=MUON_BACKEND_STEPS)
    for g in opt_muon.param_groups:
        g["base_lr"] = MATRIX_LR
    opt_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": SCALAR_LR, "base_lr": SCALAR_LR}],
                                    betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = TIME_BUDGET * 1000.0

    def lr_mul(step, elapsed_ms):
        if WARMDOWN_ITERS <= 0:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = WARMDOWN_ITERS * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup (compile paths)
    if WARMUP_STEPS > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        init_opt_states = [copy.deepcopy(o.state_dict()) for o in optimizers]
        compiled_model.train()
        for ws in range(WARMUP_STEPS):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                per_rank = TRAIN_BATCH_TOKENS // grad_accum_steps + 1
                chunk = stream.take(per_rank)
                local = chunk.to(dtype=torch.int64)
                x = local[:-1].reshape(-1, TRAIN_SEQ_LEN).to(device)
                y = local[1:].reshape(-1, TRAIN_SEQ_LEN).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = compiled_model(x, y)
                (loss / grad_accum_steps).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
        model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt_states):
            o.load_state_dict(s)
        zero_grad_all()
        stream = TokenStream()  # reset data

    # Training loop
    training_time_ms = 0.0
    stop_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (stop_step is not None and step >= stop_step)
        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for ms in range(grad_accum_steps):
            per_rank = TRAIN_BATCH_TOKENS // grad_accum_steps + 1
            chunk = stream.take(per_rank)
            local = chunk.to(dtype=torch.int64)
            x = local[:-1].reshape(-1, TRAIN_SEQ_LEN).to(device)
            y = local[1:].reshape(-1, TRAIN_SEQ_LEN).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled_model(x, y)
            train_loss += loss.detach()
            (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0) if MUON_MOMENTUM_WARMUP_STEPS > 0 else 1.0
        mom = (1 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
        for g in opt_muon.param_groups:
            g["momentum"] = mom

        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        for o in optimizers:
            o.step()
        # Decoupled weight decay for Muon
        with torch.no_grad():
            for p in matrix_params:
                p.mul_(1.0 - MUON_WEIGHT_DECAY * opt_muon.param_groups[0]["lr"])
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if step <= 10 or step % 100 == 0:
            print(f"step:{step} train_loss:{train_loss.item():.4f} time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.1f}ms")

        if stop_step is None and approx_ms >= max_wallclock_ms:
            stop_step = step

    torch.cuda.synchronize()
    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    peak_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)

    # Pre-quant eval
    pre_val_loss, pre_val_bpb = evaluate_model(model, device, val_tokens, TRAIN_SEQ_LEN)
    print(f"pre_quant_val_loss: {pre_val_loss:.4f}")
    print(f"pre_quant_val_bpb: {pre_val_bpb:.4f}")

    # Artifact size check
    size_info = compute_artifact_size(model, code_path=__file__)
    print(f"artifact_bytes: {size_info['artifact_bytes']}")
    print(f"compressed_model_bytes: {size_info['compressed_model_bytes']}")
    print(f"code_bytes: {size_info['code_bytes']}")
    print(f"artifact_valid: {size_info['valid']}")

    # Quantized roundtrip eval
    rt_info = save_and_reload_quantized(model, code_path=__file__)

    if EVAL_STRIDE > 0:
        compiled_logits = torch.compile(model.forward_logits, dynamic=False)
        # Warmup compile
        model.eval()
        warmup_x = torch.zeros(EVAL_BATCH_SEQS, TRAIN_SEQ_LEN, dtype=torch.int64, device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = compiled_logits(warmup_x)
        q_val_loss, q_val_bpb = evaluate_sliding_window(
            compiled_logits, device, val_tokens, TRAIN_SEQ_LEN, stride=EVAL_STRIDE, batch_seqs=EVAL_BATCH_SEQS)
    else:
        q_val_loss, q_val_bpb = evaluate_model(model, device, val_tokens, TRAIN_SEQ_LEN)

    # Print final summary
    print("---")
    print(f"val_bpb: {q_val_bpb:.6f}")
    print(f"pre_quant_val_bpb: {pre_val_bpb:.6f}")
    print(f"artifact_bytes: {rt_info['artifact_bytes']}")
    print(f"compressed_model_bytes: {rt_info['compressed_model_bytes']}")
    print(f"code_bytes: {rt_info['code_bytes']}")
    print(f"model_params: {n_params}")
    print(f"num_steps: {step}")
    print(f"step_avg_ms: {training_time_ms / max(step, 1):.2f}")
    print(f"training_time_ms: {training_time_ms:.0f}")
    print(f"peak_memory_mb: {peak_mem}")


if __name__ == "__main__":
    main()
