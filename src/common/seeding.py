import hashlib


def derive_seed(base_seed: int, repetition: int, stage: str) -> int:
    payload = f"{base_seed}\0{repetition}\0{stage}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
