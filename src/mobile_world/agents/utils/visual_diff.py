"""Deterministic visual diff primitive for action-effect detection.

Used by the runner to compute a binary "did the screen change" signal between
consecutive screenshots. The signal is used to drive a non-LLM trigger
condition for the checker (catches mode-collapse / no-op action loops where
the agent's actions are not producing any visible UI change).

Design rationale (see design doc 2026-05-22):
- Pure deterministic computation, no LLM call, no model dependency.
- Cross-model invariant: same screenshot pair always yields same result
  regardless of which agent model is consuming downstream.
- ~10ms per screenshot pair (1080x2400 → 9x8 Lanczos downsample + 64-bit xor).
"""

from PIL import Image


def compute_dhash(image: Image.Image, hash_size: int = 8) -> list[int]:
    """Compute a dHash signature of an image.

    Returns a list of (hash_size * hash_size) bits (0 or 1). With the default
    hash_size=8, the output is a 64-bit signature.

    Algorithm:
    1. Convert to grayscale.
    2. Downsample to (hash_size + 1) x hash_size using Lanczos resampling.
    3. For each row, compare adjacent pixels left vs right; emit 1 if left
       > right else 0.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits: list[int] = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append(1 if left > right else 0)
    return bits


def hamming_distance(h1: list[int], h2: list[int]) -> int:
    """Count differing bits between two dHash signatures.

    Range is 0 (identical) to len(h1) (fully inverted). For 64-bit signatures
    the range is 0-64.
    """
    return sum(b1 != b2 for b1, b2 in zip(h1, h2))


def is_no_change(
    h1: list[int],
    h2: list[int],
    threshold: int = 4,
) -> bool:
    """Return True if two screenshots are visually near-identical.

    A hamming distance ≤ threshold means at most `threshold` out of 64 hash
    bits differ — tolerant of status-bar time updates, anti-aliasing noise,
    and small animations, but sensitive to real UI changes (page switches,
    modals, list scrolls, button highlights, etc.).
    """
    return hamming_distance(h1, h2) <= threshold
