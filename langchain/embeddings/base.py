"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import Field, validator, BaseModel

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


class Embeddings(BaseModel, ABC):
    """Interface for embedding models."""

    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """
        If callback manager is None, set it.
        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
