import hashlib


def derive_seed(
    repetition_seed: int,
    prompt: str,
    stage: str,
) -> int:
    payload = f"{repetition_seed}\0{prompt}\0{stage}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
