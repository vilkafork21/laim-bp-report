"""
Pipeline 5. Извлечение описания архитектуры ML-решения из отчёта о разработке.

Возвращает свободный текст, описывающий путь от входных до выходных данных
с кратким описанием каждого модуля пайплайна.
"""

import time
import logging
from typing import Optional

from ..config import DEFAULT_GENERATION_CONFIG, get_chat_model, create_prompt_template
from ..prompts import ML_ARCHITECTURE_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_response, log_llm_error

logger = logging.getLogger(__name__)


class MlArchitectureExtractor:
    """
    Pipeline 5. Извлекает описание архитектуры ML-решения из отчёта о разработке.

    Возвращает строку со свободным текстом, описывающую путь от входных
    до выходных данных с кратким описанием каждого модуля.
    Пригодна для передачи в BusinessProcessVisualizer.visualize()
    в качестве аргумента ml_architecture.

    Example:
        >>> extractor = MlArchitectureExtractor(
        ...     generation_config={'model': 'GigaChat-2-Max', 'temperature': 0.3, 'max_tokens': 2048}
        ... )
        >>> architecture = extractor.extract(development_report)
        >>> # 'Входные данные: текстовые диалоги из CRM...'
    """

    def __init__(
        self,
        generation_config: Optional[dict] = None,
    ):
        """
        Args:
            generation_config: Словарь гиперпараметров генерации, распаковывается
                               в конструктор GigaChat. По умолчанию — DEFAULT_GENERATION_CONFIG.
        """
        _config = generation_config if generation_config is not None else DEFAULT_GENERATION_CONFIG
        self.generation_config = _config

        chat_model = get_chat_model(_config)
        self.prompt = create_prompt_template(ML_ARCHITECTURE_SYSTEM_PROMPT)
        self.chain = self.prompt | chat_model

    def extract(self, development_report: str) -> str:
        """
        Извлекает описание архитектуры ML-решения из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Строка с описанием архитектуры (свободный текст).
            При ошибке возвращает строку с сообщением об ошибке.
        """
        log_llm_request(
            logger, "MlArchitectureExtractor", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            result = self.chain.invoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "MlArchitectureExtractor", result, _elapsed)
            text = result.content if hasattr(result, 'content') else str(result)
            return text.strip()
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "MlArchitectureExtractor", e, _elapsed)
            return f"Ошибка при извлечении архитектуры: {e}"

    async def aextract(self, development_report: str) -> str:
        """
        Асинхронно извлекает описание архитектуры ML-решения из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Строка с описанием архитектуры (свободный текст).
        """
        log_llm_request(
            logger, "MlArchitectureExtractor (async)", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            result = await self.chain.ainvoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "MlArchitectureExtractor (async)", result, _elapsed)
            text = result.content if hasattr(result, 'content') else str(result)
            return text.strip()
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "MlArchitectureExtractor (async)", e, _elapsed)
            return f"Ошибка при извлечении архитектуры: {e}"
