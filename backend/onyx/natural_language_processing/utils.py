import os
from abc import ABC
from abc import abstractmethod
from copy import copy

from tokenizers import Encoding  # type: ignore
from tokenizers import Tokenizer  # type: ignore
from transformers import logging as transformer_logging  # type:ignore

from onyx.configs.model_configs import DOC_EMBEDDING_CONTEXT_SIZE
from onyx.configs.model_configs import DOCUMENT_ENCODER_MODEL
from onyx.context.search.models import InferenceChunk
from onyx.utils.logger import setup_logger
from shared_configs.enums import EmbeddingProvider

TRIM_SEP_PAT = "\n... {n} tokens removed...\n"

logger = setup_logger()
transformer_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def tokenize(self, string: str) -> list[str]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass


class TiktokenTokenizer(BaseTokenizer):
    _instances: dict[str, "TiktokenTokenizer"] = {}

    def __new__(cls, model_name: str) -> "TiktokenTokenizer":
        if model_name not in cls._instances:
            cls._instances[model_name] = super(TiktokenTokenizer, cls).__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        if not hasattr(self, "encoder"):
            import tiktoken

            self.encoder = tiktoken.encoding_for_model(model_name)

    def encode(self, string: str) -> list[int]:
        # this ignores special tokens that the model is trained on, see encode_ordinary for details
        return self.encoder.encode_ordinary(string)

    def tokenize(self, string: str) -> list[str]:
        encoded = self.encode(string)
        decoded = [self.encoder.decode([token]) for token in encoded]

        if len(decoded) != len(encoded):
            logger.warning(
                f"OpenAI tokenized length {len(decoded)} does not match encoded length {len(encoded)} for string: {string}"
            )

        return decoded

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.encoder: Tokenizer = Tokenizer.from_pretrained(model_name)

    def _safer_encode(self, string: str) -> Encoding:
        """
        Encode a string using the HuggingFaceTokenizer, but if it fails,
        encode the string as ASCII and decode it back to a string. This helps
        in cases where the string has weird characters like \udeb4.
        """
        try:
            return self.encoder.encode(string, add_special_tokens=False)
        except Exception:
            return self.encoder.encode(
                string.encode("ascii", "ignore").decode(), add_special_tokens=False
            )

    def encode(self, string: str) -> list[int]:
        # this returns no special tokens
        return self._safer_encode(string).ids

    def tokenize(self, string: str) -> list[str]:
        return self._safer_encode(string).tokens

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


class OpenAITokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self._delegate = TiktokenTokenizer(model_name)

    def encode(self, string: str) -> list[int]:
        return self._delegate.encode(string)

    def tokenize(self, string: str) -> list[str]:
        return self._delegate.tokenize(string)

    def decode(self, tokens: list[int]) -> str:
        return self._delegate.decode(tokens)


class CohereTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self._delegate = HuggingFaceTokenizer(model_name)

    def encode(self, string: str) -> list[int]:
        return self._delegate.encode(string)

    def tokenize(self, string: str) -> list[str]:
        return self._delegate.tokenize(string)

    def decode(self, tokens: list[int]) -> str:
        return self._delegate.decode(tokens)


_TOKENIZER_CACHE: dict[tuple[EmbeddingProvider | None, str | None], BaseTokenizer] = {}


def _check_tokenizer_cache(
    model_provider: EmbeddingProvider | None, model_name: str | None
) -> BaseTokenizer:
    global _TOKENIZER_CACHE
    id_tuple = (model_provider, model_name)

    if id_tuple not in _TOKENIZER_CACHE:
        tokenizer = None

        if model_name:
            tokenizer = _try_initialize_tokenizer(model_name, model_provider)

        if not tokenizer:
            logger.info(
                f"Falling back to default embedding model tokenizer: {DOCUMENT_ENCODER_MODEL}"
            )
            tokenizer = HuggingFaceTokenizer(DOCUMENT_ENCODER_MODEL)

        _TOKENIZER_CACHE[id_tuple] = tokenizer

    return _TOKENIZER_CACHE[id_tuple]


def _try_initialize_tokenizer(
    model_name: str, model_provider: EmbeddingProvider | None
) -> BaseTokenizer | None:
    tokenizer: BaseTokenizer | None = None

    if model_provider is not None:
        # Try using TiktokenTokenizer first if model_provider exists
        try:
            tokenizer = TiktokenTokenizer(model_name)
            logger.info(f"Initialized TiktokenTokenizer for: {model_name}")
            return tokenizer
        except Exception as tiktoken_error:
            logger.debug(
                f"TiktokenTokenizer not available for model {model_name}: {tiktoken_error}"
            )
    else:
        # If no provider specified, try HuggingFaceTokenizer
        try:
            tokenizer = HuggingFaceTokenizer(model_name)
            logger.info(f"Initialized HuggingFaceTokenizer for: {model_name}")
            return tokenizer
        except Exception as hf_error:
            logger.warning(
                f"Failed to initialize HuggingFaceTokenizer for {model_name}: {hf_error}"
            )

    # If both initializations fail, return None
    return None


# This will be used by get_tokenizer to determine the actual provider.
DOCUMENT_ENCODER_MODEL_FROM_CONFIG = DOCUMENT_ENCODER_MODEL

# Global default tokenizer, initialized lazily and safely.
_DEFAULT_TOKENIZER: BaseTokenizer | None = None
_DEFAULT_TOKENIZER_MODEL_NAME: str | None = None


def get_tokenizer(
    model_name: str,
    provider_type: EmbeddingProvider | None = None,
) -> BaseTokenizer:
    global _DEFAULT_TOKENIZER, _DEFAULT_TOKENIZER_MODEL_NAME

    # 1. Explicit Provider Handling (Preferred)
    if provider_type == EmbeddingProvider.OPENAI:
        return OpenAITokenizer(model_name=model_name)
    if provider_type == EmbeddingProvider.COHERE:
        return CohereTokenizer(model_name=model_name)
    # Add other explicit provider checks here if necessary

    # 2. Model Name Based Heuristics (if provider_type is None)
    if provider_type is None:
        if model_name.startswith("text-embedding-"):  # OpenAI
            return OpenAITokenizer(model_name=model_name)
        if model_name.startswith("embed-"):  # Cohere (basic check)
            return CohereTokenizer(model_name=model_name)
        # Add other model_name based heuristics here

    # 3. Fallback to HuggingFace Tokenizer (_DEFAULT_TOKENIZER logic)
    # This path is taken if provider_type is None or an unhandled/HF provider,
    # AND model_name heuristics didn't match.
    # We assume model_name at this point *should* be a HuggingFace model.
    
    # If _DEFAULT_TOKENIZER is not initialized, or is for a different model
    if _DEFAULT_TOKENIZER is None or _DEFAULT_TOKENIZER_MODEL_NAME != model_name:
        try:
            logger.debug(f"Initializing/Re-initializing default HuggingFace tokenizer for: {model_name}")
            _DEFAULT_TOKENIZER = HuggingFaceTokenizer(model_name)
            _DEFAULT_TOKENIZER_MODEL_NAME = model_name
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace tokenizer for '{model_name}'. Error: {e}")
            # If this fails, it's a critical error for this path.
            # Consider what the true 'system default safe' HF tokenizer should be if model_name itself fails.
            # Perhaps DEFAULT_DOCUMENT_ENCODER_MODEL from model_configs.py (i.e., Nomic)
            # could be a final fallback if model_name is not a valid HF model.
            # For now, re-raise.
            raise RuntimeError(
                f"Failed to initialize HuggingFace tokenizer for the model: '{model_name}'. "
                "Ensure it's a valid HuggingFace model identifier if no specific provider is set."
            ) from e
            
    return _DEFAULT_TOKENIZER


def tokenizer_trim_content(
    content: str, desired_length: int, tokenizer: BaseTokenizer
) -> str:
    tokens = tokenizer.encode(content)
    if len(tokens) <= desired_length:
        return content

    return tokenizer.decode(tokens[:desired_length])


def tokenizer_trim_middle(
    tokens: list[int], desired_length: int, tokenizer: BaseTokenizer
) -> str:
    if len(tokens) <= desired_length:
        return tokenizer.decode(tokens)
    sep_str = TRIM_SEP_PAT.format(n=len(tokens) - desired_length)
    sep_tokens = tokenizer.encode(sep_str)
    slice_size = (desired_length - len(sep_tokens)) // 2
    assert slice_size > 0, "Slice size is not positive, desired length is too short"
    return (
        tokenizer.decode(tokens[:slice_size])
        + sep_str
        + tokenizer.decode(tokens[-slice_size:])
    )


def tokenizer_trim_chunks(
    chunks: list[InferenceChunk],
    tokenizer: BaseTokenizer,
    max_chunk_toks: int = DOC_EMBEDDING_CONTEXT_SIZE,
) -> list[InferenceChunk]:
    new_chunks = copy(chunks)
    for ind, chunk in enumerate(new_chunks):
        new_content = tokenizer_trim_content(chunk.content, max_chunk_toks, tokenizer)
        if len(new_content) != len(chunk.content):
            new_chunk = copy(chunk)
            new_chunk.content = new_content
            new_chunks[ind] = new_chunk
    return new_chunks
