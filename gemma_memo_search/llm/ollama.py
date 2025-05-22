"""Ollama client for Gemma LLM integration."""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult

from ..config import LLM_CONFIG

logger = logging.getLogger(__name__)


class GemmaLLM(LLM):
    """Wrapper for Gemma LLM using Ollama."""

    model: str = LLM_CONFIG["model"]
    host: str = LLM_CONFIG["host"]
    port: int = LLM_CONFIG["port"]
    temperature: float = LLM_CONFIG["temperature"]
    top_p: float = LLM_CONFIG["top_p"]
    max_tokens: int = LLM_CONFIG["max_tokens"]
    ollama_client: Optional[Ollama] = None

    def __init__(
        self,
        model: str = LLM_CONFIG["model"],
        host: str = LLM_CONFIG["host"],
        port: int = LLM_CONFIG["port"],
        temperature: float = LLM_CONFIG["temperature"],
        top_p: float = LLM_CONFIG["top_p"],
        max_tokens: int = LLM_CONFIG["max_tokens"],
        **kwargs: Any,
    ):
        """Initialize the Gemma LLM.

        Args:
            model: Gemma model to use (gemma:3b, gemma:7b)
            host: Ollama server host
            port: Ollama server port
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model = model
        self.host = host
        self.port = port
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize Ollama client
        self._initialize_ollama()

    def _initialize_ollama(self) -> None:
        """Initialize the Ollama client."""
        try:
            self.ollama_client = Ollama(
                model=self.model,
                base_url=f"http://{self.host}:{self.port}",
                temperature=self.temperature,
                top_p=self.top_p,
                num_predict=self.max_tokens,
            )
            logger.info(f"Initialized Ollama client for model: {self.model}")
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemma_ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Ollama API to generate text.

        Args:
            prompt: The prompt to generate text from
            stop: A list of strings to stop generation when encountered
            run_manager: Callback manager
            **kwargs: Additional keyword arguments

        Returns:
            Generated text
        """
        if self.ollama_client is None:
            self._initialize_ollama()

        try:
            # Pass through to Ollama client
            response = self.ollama_client._call(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return response
        except Exception as e:
            error_msg = f"Error generating text with Ollama: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously call the Ollama API to generate text.

        Args:
            prompt: The prompt to generate text from
            stop: A list of strings to stop generation when encountered
            run_manager: Callback manager
            **kwargs: Additional keyword arguments

        Returns:
            Generated text
        """
        if self.ollama_client is None:
            self._initialize_ollama()

        try:
            # Pass through to Ollama client
            response = await self.ollama_client._acall(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return response
        except Exception as e:
            error_msg = f"Error generating text with Ollama: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"