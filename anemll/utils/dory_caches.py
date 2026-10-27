"""Dory KV-cache layout, ring-buffer writes and mask construction.

Spec D4 (`last_loop`), D5 (ANE-safe cache writes), D7 (cache memory).

Cache inventory (authoritative: llama.cpp dory.cpp:70-79, math-dory.md G2/G3)
---------------------------------------------------------------------------
Under `last_loop` there is one cache per **physical attention block**: 36 of
them, 27 sliding and 9 full.  Only the 20 recurrent-zone attention caches are
written once per loop pass; the input/output zone ones are written once per
token.  The cache shape is therefore independent of the loop count, which is
what lets the host choose the loop count at run time.

Rows
----
* full caches    : `context_length` rows (24,576 for the shipped context).
* sliding caches  : `PAD(min(ctx, sliding_window + batch_size), 256)` rows --
  the same sizing llama.cpp uses (llama-kv-cache-iswa.cpp:69-105).  A ring of
  exactly `sliding_window` rows could not hold "window + current batch".

Writes are **single-row** (decode) or **n-row slice** (prefill) updates of the
state buffer.  The sliding window advances by *index arithmetic*
(`cache_pos = current_pos % rows`), never by shifting or rewriting the cache:
attention is a softmax over a *set* of keys, so the ring's physical order does
not matter and no rotate copy is needed.  This is what spec D5's "never a
mask-multiply rewrite of the cache" costs at window 1,024 x 27 layers.

Masks
-----
The mask is an additive float16 input tensor of width
`context_length + sliding_rows`: columns `[0:context_length)` carry the
full-attention causal mask, columns `[context_length:]` carry the
sliding-window mask in **ring-slot order**.  Both are built on the host by
:func:`build_causal_mask`; the model slices them apart per layer type.
"""

from __future__ import annotations

import torch

# The 8-block period of Dory's physical layout (CLAUDE/GGUF `layer_pattern`):
# sliding, mlp, sliding, mlp, sliding, mlp, full, mlp -- repeated 9 times.
PERIOD = 8
FULL_ATTENTION_OFFSET = 6
SLIDING_ATTENTION_OFFSETS = (0, 2, 4)

NUM_BLOCKS = 72
INPUT_BLOCK_COUNT = 16
RECURRENT_BLOCK_COUNT = 40
OUTPUT_BLOCK_COUNT = 16
SLIDING_WINDOW = 1024

NEG_INF = float("-inf")


def block_is_full_attention(block: int) -> bool:
    """True for the 9 full-attention blocks (theta = 1e7)."""
    return block % PERIOD == FULL_ATTENTION_OFFSET


def block_is_sliding_attention(block: int) -> bool:
    """True for the 27 sliding-window blocks (theta = 1e4, window 1024)."""
    return block % PERIOD in SLIDING_ATTENTION_OFFSETS


def block_is_attention(block: int) -> bool:
    return block_is_full_attention(block) or block_is_sliding_attention(block)


def block_is_mlp(block: int) -> bool:
    return not block_is_attention(block)


def full_attention_blocks():
    return tuple(b for b in range(NUM_BLOCKS) if block_is_full_attention(b))


def sliding_attention_blocks():
    return tuple(b for b in range(NUM_BLOCKS) if block_is_sliding_attention(b))


def cache_slots():
    """Map physical block -> (kind, slot index).

    Returns `(full_slots, sliding_slots)`, dicts physical-block -> slot.
    There are 9 full and 27 sliding slots (math-dory.md G2).
    """
    full, sliding = {}, {}
    for block in range(NUM_BLOCKS):
        if block_is_full_attention(block):
            full[block] = len(full)
        elif block_is_sliding_attention(block):
            sliding[block] = len(sliding)
    return full, sliding


def _pad_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def full_cache_rows(context_length: int) -> int:
    """Rows of one full-attention cache (K and V each)."""
    return int(context_length)


def sliding_cache_rows(context_length: int, sliding_window: int = SLIDING_WINDOW,
                       batch_size: int = 64) -> int:
    """Rows of one sliding cache: `PAD(min(ctx, window + batch), 256)`.

    llama.cpp sizes its SWA cache exactly this way (llama-kv-cache-iswa.cpp:69-105);
    the extra `batch_size` rows are what lets a full prefill batch be written
    before its oldest rows are recycled.  With `batch_size` 64 the write offset
    is always a multiple of 64, so a batch never straddles the ring seam.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return _pad_to(min(int(context_length), int(sliding_window) + int(batch_size)), 256)


def cache_shapes(context_length: int, num_kv_heads: int, head_dim: int,
                 sliding_window: int = SLIDING_WINDOW, batch_size: int = 64):
    """Shapes of the two state buffers, K and V interleaved along dim 0."""
    full_slots, sliding_slots = cache_slots()
    full = (2 * len(full_slots), num_kv_heads, full_cache_rows(context_length), head_dim)
    sliding = (2 * len(sliding_slots), num_kv_heads,
               sliding_cache_rows(context_length, sliding_window, batch_size), head_dim)
    return full, sliding


def write_cache_rows(cache: torch.Tensor, slot: int, key: torch.Tensor, value: torch.Tensor,
                     cache_pos, num_slots: int) -> None:
    """Write one row (decode) or a row slice (prefill) into a K/V state buffer.

    `cache` is the `(2 * num_slots, num_kv_heads, rows, head_dim)` buffer,
    K in slice `[slot]` and V in slice `[slot + num_slots]`.
    `cache_pos` may be a python int or a 0-d / 1-element int tensor (the
    traced form reads it from an int32 graph input).
    `key` / `value` are `(1, num_kv_heads, n, head_dim)` with `n` the number
    of rows written -- exactly the slice written, never the whole cache.
    """
    seq = key.shape[2]
    cache[slot:slot + 1, :, cache_pos:cache_pos + seq, :] = key.to(cache.dtype)
    cache[slot + num_slots:slot + num_slots + 1, :, cache_pos:cache_pos + seq, :] = \
        value.to(cache.dtype)


def read_cache_slot(cache: torch.Tensor, slot: int, num_slots: int, rows: int):
    """Read one cache slot as `(1, num_kv_heads, rows, head_dim)` K and V."""
    return (cache[slot:slot + 1, :, :rows, :],
            cache[slot + num_slots:slot + num_slots + 1, :, :rows, :])


def reset_caches(*caches: torch.Tensor) -> None:
    """Zero the state buffers -- what a loop-count change must do (spec D4)."""
    for cache in caches:
        cache.zero_()


def full_causal_mask(start: int, q_len: int, rows: int, dtype=torch.float16) -> torch.Tensor:
    """Additive causal mask for the full caches: 0 at `col <= query_pos`, -inf above.

    Shape `(1, 1, q_len, rows)`.  Rows past the current position are masked by
    construction (the cache is zero there but must not be attended to).
    """
    query_pos = torch.arange(start, start + q_len, dtype=torch.int64).view(q_len, 1)
    cols = torch.arange(rows, dtype=torch.int64).view(1, rows)
    allowed = cols <= query_pos
    mask = torch.zeros(q_len, rows, dtype=torch.float32)
    mask.masked_fill_(~allowed, NEG_INF)
    return mask.view(1, 1, q_len, rows).to(dtype)


def sliding_ring_mask(start: int, q_len: int, rows: int, sliding_window: int = SLIDING_WINDOW,
                      dtype=torch.float16) -> torch.Tensor:
    """Additive mask for a sliding cache addressed as a ring, in ring-slot order.

    Slot `j` holds the newest position `q_j <= last` with `q_j % rows == j`
    (`last = start + q_len - 1`, the batch's newest row).  A query at `pq`
    may see slot `j` iff the token there is in its window:
    `0 <= pq - q_j <= sliding_window - 1` (inclusive window, math-dory.md 5.1),
    and `q_j >= 0` (the slot has been written at all).
    """
    last = start + q_len - 1
    if last < 0:
        raise ValueError("start + q_len must be >= 1")
    slots = torch.arange(rows, dtype=torch.int64).view(1, rows)
    offset = torch.remainder(last - slots, rows)          # (1, rows), always >= 0
    tok_pos = last - offset                                # position currently in slot j
    query_pos = torch.arange(start, start + q_len, dtype=torch.int64).view(q_len, 1)
    age = query_pos - tok_pos                              # (q_len, rows)
    allowed = (tok_pos >= 0) & (age >= 0) & (age <= sliding_window - 1)
    mask = torch.zeros(q_len, rows, dtype=torch.float32)
    mask.masked_fill_(~allowed, NEG_INF)
    return mask.view(1, 1, q_len, rows).to(dtype)


def build_causal_mask(start: int, q_len: int, context_length: int, sliding_rows: int,
                      sliding_window: int = SLIDING_WINDOW, dtype=torch.float16) -> torch.Tensor:
    """The single mask input: full-attention part then sliding part, concatenated.

    Shape `(1, 1, q_len, context_length + sliding_rows)`.
    """
    return torch.cat([
        full_causal_mask(start, q_len, context_length, dtype=dtype),
        sliding_ring_mask(start, q_len, sliding_rows, sliding_window, dtype=dtype),
    ], dim=-1)


# ---------------------------------------------------------------------------
# The 24,576-row full-attention caches: one state, or two tiles (spec D7)
# ---------------------------------------------------------------------------

#: Spec D7's fallback for a full-attention cache that falls off the ANE at its
#: full width: 16,384 + 8,192 rows, the two attention partials merged with a
#: log-sum-exp combine.  16,384 is the documented ANE axis guidance, 8,192 the
#: rest of a 24,576-token context.
FULL_TILES_24K = (16384, 8192)


def full_cache_tiles(context_length: int, tiles=None) -> tuple:
    """Row counts of the full-attention cache tiles; they sum to the context.

    tiles=None (or a single tile equal to the context) is the shipped shape: one
    cache of context_length rows.  Two tiles are the spec's fallback.  Any other
    layout is refused rather than silently approximated -- tiles that do not add
    up to the context would lose tokens.
    """
    ctx = int(context_length)
    if ctx <= 0:
        raise ValueError("context_length must be positive")
    if tiles is None:
        return (ctx,)
    if isinstance(tiles, int):
        tiles = (tiles,)
    rows = tuple(int(t) for t in tiles)
    if len(rows) not in (1, 2):
        raise ValueError(f"a full-attention cache is one cache or the spec's two "
                         f"tiles, got {len(rows)} tiles {rows}")
    if any(r <= 0 for r in rows):
        raise ValueError(f"tile rows must be positive, got {rows}")
    if sum(rows) != ctx:
        raise ValueError(f"tiles {rows} sum to {sum(rows)}, not the context length "
                         f"{ctx}: a layout that does not add up would drop tokens")
    return rows


def full_tile_offsets(tiles) -> tuple:
    """Absolute position where each tile starts: (0, 16384) at 24,576 rows."""
    offsets, start = [], 0
    for rows in tiles:
        offsets.append(start)
        start += int(rows)
    return tuple(offsets)


def full_cache_shapes(context_length: int, num_kv_heads: int, head_dim: int,
                      tiles=None, num_slots: int = None) -> list:
    """One (2 * slots, kv, rows, head_dim) shape per full-attention tile."""
    if num_slots is None:
        num_slots = len(cache_slots()[0])
    return [(2 * int(num_slots), int(num_kv_heads), int(rows), int(head_dim))
            for rows in full_cache_tiles(context_length, tiles)]


def sliding_phase(pos: int, sliding_window: int = SLIDING_WINDOW) -> str:
    """Which sliding-cache write applies at pos (Gemma 3 prior art).

    'fill' while the window is still filling (the row is written at pos);
    'rotate' from then on (the cache is full, so the oldest row is dropped and
    the new one appended).  The ring cache of this module needs neither decision
    -- the write offset is pos % rows in both phases, which is why a ring costs
    no copy -- but the ANEMLL prior art names the two functions this way,
    rotate_cache_rows implements the physical rotate, and the test that pins
    ring == rotate uses both.
    """
    pos = int(pos)
    if pos < 0:
        raise ValueError("pos must be >= 0")
    return "fill" if pos < int(sliding_window) else "rotate"


def rotate_cache_rows(cache: torch.Tensor, slot: int, key: torch.Tensor,
                      value: torch.Tensor, rows: int, num_slots: int) -> None:
    """Shift-left + append (Gemma 3 _update_kv_local), the physical rotate.

    Positive constants only: torch.narrow with start = the number of new rows and
    length = rows - that, then torch.cat.  cache[:-1] = cache[1:] and negative
    slices are what break under torch.jit.trace and on the ANE (gemma3_model.py:
    1035-1039), so neither appears here.
    """
    new_rows = int(key.shape[2])
    rows = int(rows)
    if not 0 < new_rows <= rows:
        raise ValueError(f"cannot rotate {new_rows} new rows into a {rows}-row cache")
    for idx, new in ((slot, key), (slot + num_slots, value)):
        current = cache[idx:idx + 1, :, :rows, :]
        tail = torch.narrow(current, 2, new_rows, rows - new_rows)
        shifted = torch.cat([tail, new.to(cache.dtype)], dim=2)
        cache[idx:idx + 1, :, :rows, :] = shifted


def merge_attention_partials(partials):
    """Log-sum-exp combine of the attention partials of a tiled cache.

    partials is a list of (logits, values) pairs for one query block: logits
    (B, KV, rep*Q, rows_t) already carrying that tile's mask (so columns past the
    current position are -inf) and values (1, KV, rows_t, head_dim).  Returns
    (out, total_weights) with out (B, KV, rep*Q, head_dim).

    Attention is a softmax over a *set* of keys, so splitting the set in two and
    subtracting the maximum over both tiles is exact; a tile whose columns are
    all masked contributes exp(-inf - m) = 0, which is why the young positions --
    where the second tile is entirely in the future -- need no special case.
    This is spec D7's combine, and also the numerically stable form of an
    ordinary softmax.
    """
    if not partials:
        raise ValueError("merge_attention_partials needs at least one partial")
    maximum = None
    for logits, _ in partials:
        tile_max = logits.amax(dim=-1, keepdim=True)
        maximum = tile_max if maximum is None else torch.maximum(maximum, tile_max)
    out = None
    total = None
    for logits, values in partials:
        weights = torch.exp(logits - maximum)
        tile_total = weights.sum(dim=-1, keepdim=True)
        contribution = torch.matmul(weights, values)
        out = contribution if out is None else out + contribution
        total = tile_total if total is None else total + tile_total
    return out / total, total


def build_tile_masks(start: int, q_len: int, tiles, sliding_rows: int,
                     sliding_window: int = SLIDING_WINDOW, dtype=torch.float16) -> list:
    """One mask per full-attention tile, then the sliding mask, as a list.

    Tile i owns absolute positions [offset, offset + rows), so its column j is
    position offset + j and the causal test is j <= query_pos - offset: the same
    full_causal_mask with the tile's offset taken off the query row.  A query
    younger than the tile finds every column masked, which is what the
    log-sum-exp merge expects.
    """
    rows_list = full_cache_tiles(int(sum(tiles)), tiles)
    masks = [full_causal_mask(int(start) - offset, int(q_len), rows, dtype=dtype)
             for offset, rows in zip(full_tile_offsets(rows_list), rows_list)]
    masks.append(sliding_ring_mask(int(start), int(q_len), int(sliding_rows),
                                   int(sliding_window), dtype=dtype))
    return masks


def cache_positions(start: int, q_len: int, rows: int):
    """Ring positions of the rows a batch of `q_len` at `start` writes.

    Returns a list of ints; with `q_len` a multiple of the ubatch and
    `rows % batch_size == 0` these never straddle the seam, so the write is one
    contiguous slice (see `sliding_cache_rows`).
    """
    positions = [ (start + i) % rows for i in range(q_len) ]
    contiguous = all(positions[i] + 1 == positions[i + 1] for i in range(len(positions) - 1))
    if not contiguous:
        raise ValueError(f"ring write for start={start} q_len={q_len} rows={rows} "
                         "would straddle the cache seam")
    return positions[0]
