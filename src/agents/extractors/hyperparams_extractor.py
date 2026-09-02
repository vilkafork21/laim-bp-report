"""
Pipeline 3. Извлечение гиперпараметров генерации LLM из отчёта о разработке.
"""

import time
import logging
from typing import Optional, Dict, List

from pydantic import BaseModel, Field

from ..config import DEFAULT_GENERATION_CONFIG, get_chat_model, create_prompt_template
from ..prompts import HYPERPARAMS_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_response, log_llm_error

logger = logging.getLogger(__name__)


class HyperparamItem(BaseModel):
    """Один гиперпараметр генерации."""

    name: str = Field(description="Название гиперпараметра, как указано в отчёте.")
    value: str = Field(description="Значение гиперпараметра в виде строки.")


class HyperparamsOutput(BaseModel):
    """Список гиперпараметров генерации, извлечённых из отчёта."""

    hyperparams: List[HyperparamItem] = Field(
        description=(
            "Список гиперпараметров генерации. "
            "Пустой список, если гиперпараметры в отчёте не упомянуты."
        )
    )


class HyperparamsExtractor:
    """
    Pipeline 3. Извлекает гиперпараметры генерации LLM из отчёта о разработке.

    Возвращает словарь {название: значение}, пригодный для передачи в
    BusinessProcessVisualizer.visualize() в качестве аргумента generation_hyperparams.

    Example:
        >>> extractor = HyperparamsExtractor(
        ...     generation_config={'model': 'GigaChat-2-Max', 'temperature': 0.0, 'max_tokens': 2048}
        ... )
        >>> hyperparams = extractor.extract(development_report)
        >>> # {'temperature': '0.7', 'max_tokens': '2048', 'model': 'GigaChat-2-Max'}
    """

    def __init__(self, generation_config: Optional[dict] = None,):
        """
        Args:
            generation_config: Словарь гиперпараметров генерации, распаковывается
                               в конструктор GigaChat. По умолчанию — DEFAULT_GENERATION_CONFIG.
        """
        _config = generation_config if generation_config is not None else DEFAULT_GENERATION_CONFIG
        self.generation_config = _config

        chat_model = get_chat_model(_config)
        self.structured_model = chat_model.with_structured_output(
            HyperparamsOutput, include_raw=True
        )
        self.prompt = create_prompt_template(HYPERPARAMS_SYSTEM_PROMPT)
        self.chain = self.prompt | self.structured_model

    def extract(self, development_report: str) -> Dict[str, str]:
        """
        Извлекает гиперпараметры генерации из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Словарь {название_параметра: значение}.
            Пустой словарь, если гиперпараметры в отчёте не упомянуты.
            При ошибке возвращает словарь с ключом 'error'.
        """
        log_llm_request(
            logger, "HyperparamsExtractor", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = self.chain.invoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "HyperparamsExtractor", _raw, _elapsed)
            result: HyperparamsOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return {item.name: item.value for item in result.hyperparams}
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "HyperparamsExtractor", e, _elapsed)
            return {'error': f"Ошибка при извлечении: {e}"}

    async def aextract(self, development_report: str) -> Dict[str, str]:
        """
        Асинхронно извлекает гиперпараметры генерации из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Словарь {название_параметра: значение}.
        """
        log_llm_request(
            logger, "HyperparamsExtractor (async)", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = await self.chain.ainvoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "HyperparamsExtractor (async)", _raw, _elapsed)
            result: HyperparamsOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return {item.name: item.value for item in result.hyperparams}
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "HyperparamsExtractor (async)", e, _elapsed)
            return {'error': f"Ошибка при извлечении: {e}"}
