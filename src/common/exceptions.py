class ModelLoadException(Exception):
    """Raised when the Provider is not running."""


class ModelNotFoundException(Exception):
    """Raised when the requested LLM model does not exist."""
