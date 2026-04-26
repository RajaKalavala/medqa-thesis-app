"""Dataset loaders."""
from medqa_rag.data.loaders.medqa_loader import load_medqa
from medqa_rag.data.loaders.textbook_loader import load_textbooks

__all__ = ["load_medqa", "load_textbooks"]
