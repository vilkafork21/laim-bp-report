"""
Pipeline 2. Генерация человекочитаемой суммаризации отчёта о разработке.

Принимает на вход отчёт и словарь ключевых параметров (результат KeyPointsExtractor),
отвечает на три ключевых вопроса:
  1. Какая проблема была без модели?
  2. Какую модель построили для решения?
  3. Как модель решает эту проблему?
"""

import time
import logging
from typing import Optional, Dict

from ..config import DEFAULT_GENERATION_CONFIG, get_chat_model, create_prompt_template
from ..prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_HUMAN_TEMPLATE
from src.utils.llm_logging import log_llm_request, log_llm_response, log_llm_error

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Pipeline 2. Генерирует человекочитаемую суммаризацию отчёта о разработке.

    Результат предназначен для передачи в BusinessProcessVisualizer.visualize()
    в качестве аргумента summary.

    Example:
        >>> generator = SummaryGenerator(
        ...     generation_config={'model': 'GigaChat-2-Max', 'temperature': 0.3, 'max_tokens': 1024}
        ... )
        >>> summary = generator.generate(development_report, key_points)
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
        self.prompt = create_prompt_template(
            SUMMARY_SYSTEM_PROMPT
        )
        self.chain = self.prompt | chat_model

    @staticmethod
    def _key_points_have_errors(key_points: Dict[str, str]) -> bool:
        """Возвращает True, если хотя бы одно значение — строка-ошибка из KeyPointsExtractor."""
        return any(
            isinstance(v, str) and v.startswith('Ошибка при извлечении:')
            for v in key_points.values()
        )

    def generate(self, development_report: str, key_points: Dict[str, str]) -> str:
        """
        Генерирует суммаризацию отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.
            key_points: Словарь ключевых параметров (результат KeyPointsExtractor.extract()).
                        Если все значения — строки-ошибки, генерация не выполняется.

        Returns:
            Строка с суммаризацией на русском языке.
            При ошибке (в том числе если key_points содержит ошибки) возвращает
            сообщение об ошибке.
        """
        if self._key_points_have_errors(key_points):
            logger.warning("[SummaryGenerator] key_points contain extraction errors, skipping generation")
            return "Суммаризация недоступна: ключевые параметры не были извлечены."

        key_points_text = '\n'.join(f'  {k}: {v}' for k, v in key_points.items())
        log_llm_request(
            logger, "SummaryGenerator", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            result = self.chain.invoke('Отчет о разработке \n' + development_report + '\n Ключевые параметры \n' + key_points_text)
            _elapsed = time.time() - _start
            log_llm_response(logger, "SummaryGenerator", result, _elapsed)
            return result.content
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "SummaryGenerator", e, _elapsed)
            return f"Ошибка при генерации суммаризации: {e}"

    async def agenerate(self, development_report: str, key_points: Dict[str, str]) -> str:
        """
        Асинхронно генерирует суммаризацию отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.
            key_points: Словарь ключевых параметров (результат KeyPointsExtractor.aextract()).
                        Если все значения — строки-ошибки, генерация не выполняется.

        Returns:
            Строка с суммаризацией на русском языке.
        """
        if self._key_points_have_errors(key_points):
            logger.warning("[SummaryGenerator] key_points contain extraction errors, skipping generation")
            return "Суммаризация недоступна: ключевые параметры не были извлечены."

        key_points_text = '\n'.join(f'  {k}: {v}' for k, v in key_points.items())
        log_llm_request(
            logger, "SummaryGenerator (async)", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            result = await self.chain.ainvoke('Отчет о разработке \n' + development_report + '\n Ключевые параметры \n' + key_points_text)
            _elapsed = time.time() - _start
            log_llm_response(logger, "SummaryGenerator (async)", result, _elapsed)
            return result.content
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "SummaryGenerator (async)", e, _elapsed)
            return f"Ошибка при генерации суммаризации: {e}"
