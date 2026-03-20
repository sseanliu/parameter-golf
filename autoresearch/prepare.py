"""
Parameter Golf Autoresearch -- Fixed utilities.
DO NOT MODIFY THIS FILE. The agent modifies train.py only.

Provides: data loading, evaluation, quantization, and size checking.
"""

import glob
import io
import math
import os
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# ===========================================================================
# CONSTANTS
# ===========================================================================

TIME_BUDGET = 300  # 5 minutes wall clock for training
MAX_ARTIFACT_BYTES = 16_000_000  # 16MB hard cap
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
TRAIN_FILES = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_FILES = os.path.join(DATA_PATH, "fineweb_val_*.bin")
VOCAB_SIZE = 1024
DEFAULT_SEQ_LEN = 1024

# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(VAL_FILES))]
    if not files:
        raise FileNotFoundError(f"No val files found: {VAL_FILES}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern: str = TRAIN_FILES):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


# ===========================================================================
# EVALUATION (BPB)
# ===========================================================================

def build_sentencepiece_luts(device: torch.device):
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, VOCAB_SIZE)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_np, dtype=torch.bool, device=device),
    )


def evaluate_model(model, device, val_tokens, seq_len, batch_tokens=524_288):
    """Evaluate model and return (val_loss, val_bpb)."""
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(device)
    local_batch_seqs = batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            tok_count += n
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_bytes[tgt_ids].to(torch.int16)
            tb += (has_space[tgt_ids] & ~is_boundary[prev_ids]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()

    val_loss = (loss_sum / tok_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()
    model.train()
    return val_loss, bpt * tpb


def evaluate_sliding_window(logits_fn, device, val_tokens, seq_len, stride=64, batch_seqs=256):
    """Sliding window evaluation for better BPB."""
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(device)
    total = val_tokens.numel() - 1

    windows = []
    p = 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride)
        windows.append((p, s))
        p += stride

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for i in range(0, len(windows), batch_seqs):
            batch = windows[i:i + batch_seqs]
            bs = len(batch)
            x_list = [val_tokens[w:w + seq_len] for w, _ in batch]
            y_list = [val_tokens[w + 1:w + seq_len + 1] for w, _ in batch]
            pad = batch_seqs - bs
            if pad > 0:
                x_list.extend([x_list[-1]] * pad)
                y_list.extend([y_list[-1]] * pad)
            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x)
            for b in range(bs):
                s = batch[b][1]
                sl = logits[b, s:]
                st = y[b, s:]
                loss = F.cross_entropy(sl.float(), st, reduction="sum")
                loss_sum += loss.to(torch.float64)
                ns = st.numel()
                tok_count += ns
                prev = x[b, s:s + ns]
                tb = base_bytes[st].to(torch.int16)
                tb += (has_space[st] & ~is_boundary[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

    val_loss = (loss_sum / tok_count).item()
    bpb = val_loss / math.log(2.0) * (tok_count.item() / byte_count.item())
    return val_loss, bpb


# ===========================================================================
# QUANTIZATION + SIZE CHECKING
# ===========================================================================

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
                    "iter_attn", "iter_mlp", "iter_resid")

def quantize_state_dict_int8(state_dict):
    """Quantize model to int8+zlib. Returns (quant_obj, stats)."""
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig = {}, {}
    qmeta = {}
    stats = {"param_count": 0, "baseline_bytes": 0, "int8_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["baseline_bytes"] += t.numel() * t.element_size()

        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_bytes"] += t.numel() * t.element_size()
            continue

        # Keep embedding in fp16
        if "tok_emb" in name:
            kept = t.to(torch.float16).contiguous()
            passthrough[name] = kept
            passthrough_orig[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_bytes"] += kept.numel() * kept.element_size()
            continue

        # Small tensors or control tensors: keep as fp16/fp32
        if t.numel() <= 65536 or any(p in name for p in CONTROL_PATTERNS):
            if any(p in name for p in CONTROL_PATTERNS):
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            stats["int8_bytes"] += kept.numel() * kept.element_size()
            continue

        # Quantize to int8
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1) if t32.numel() else torch.empty(t32.shape[0])
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            scales[name] = scale.to(torch.float16).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
            scales[name] = scale

        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_bytes"] += q.numel() * q.element_size() + scales[name].numel() * scales[name].element_size()

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig:
        obj["passthrough_orig_dtypes"] = passthrough_orig
    return obj, stats


def dequantize_state_dict_int8(obj):
    """Recover float state dict from int8 quantized object."""
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = passthrough_orig.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out


def compute_artifact_size(model, code_path="train.py"):
    """Quantize model, compress, return artifact size breakdown."""
    state = dict(model.state_dict())
    quant_obj, stats = quantize_state_dict_int8(state)

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)

    code_bytes = len(Path(code_path).read_text(encoding="utf-8").encode("utf-8"))
    model_bytes = len(compressed)
    total = code_bytes + model_bytes

    return {
        "code_bytes": code_bytes,
        "compressed_model_bytes": model_bytes,
        "artifact_bytes": total,
        "param_count": stats["param_count"],
        "valid": total <= MAX_ARTIFACT_BYTES,
    }


def save_and_reload_quantized(model, code_path="train.py"):
    """Quantize, compress, decompress, reload -- full roundtrip. Returns size info."""
    state = dict(model.state_dict())
    quant_obj, _ = quantize_state_dict_int8(state)

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)

    # Roundtrip
    reloaded = torch.load(io.BytesIO(zlib.decompress(compressed)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(reloaded), strict=False)

    code_bytes = len(Path(code_path).read_text(encoding="utf-8").encode("utf-8"))
    model_bytes = len(compressed)
    return {
        "code_bytes": code_bytes,
        "compressed_model_bytes": model_bytes,
        "artifact_bytes": code_bytes + model_bytes,
    }
