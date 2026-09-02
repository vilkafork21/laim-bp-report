"""
Pipeline 1. Извлечение ключевых параметров бизнес-процесса из отчёта о разработке.
"""

import time
import logging
from typing import Optional, Dict

from pydantic import BaseModel, Field

from ..config import DEFAULT_GENERATION_CONFIG, get_chat_model, create_prompt_template
from ..prompts import KEY_POINTS_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_response, log_llm_error

logger = logging.getLogger(__name__)


class KeyPointsOutput(BaseModel):
    """Структурированный вывод — ключевые параметры бизнес-процесса."""

    target_nature: str = Field(
        description=(
            "Природа целевой переменной: сущность, которую предсказывает модель. "
            "'-' если целевая переменная отсутствует."
        )
    )
    business_task: str = Field(
        description="Краткое описание бизнес-задачи (1–2 предложения)."
    )
    ml_task: str = Field(
        description="Тип решаемой ML-задачи (классификация, NER, генерация текста и т.д.)."
    )
    model_inference_scenario: str = Field(
        description=(
            "Сценарий применения модели в бизнесе: когда запускается, "
            "на каких данных, как используются результаты."
        )
    )
    model_application_segment: str = Field(
        description="Сегмент данных, на которых будет применяться модель."
    )


class KeyPointsExtractor:
    """
    Pipeline 1. Извлекает ключевые параметры бизнес-процесса из отчёта о разработке.

    Возвращает словарь, пригодный для передачи в BusinessProcessVisualizer.visualize()
    в качестве аргумента key_points.

    Example:
        >>> extractor = KeyPointsExtractor(
        ...     generation_config={'model': 'GigaChat-2-Max', 'temperature': 0.0, 'max_tokens': 2048}
        ... )
        >>> key_points = extractor.extract(development_report)
    """

    def __init__(self, generation_config: Optional[dict] = None):
        """
        Args:
            generation_config: Словарь гиперпараметров генерации, распаковывается
                               в конструктор GigaChat. По умолчанию — DEFAULT_GENERATION_CONFIG.
        """
        _config = generation_config if generation_config is not None else DEFAULT_GENERATION_CONFIG
        self.generation_config = _config
        chat_model = get_chat_model(_config)
        self.structured_model = chat_model.with_structured_output(
            KeyPointsOutput, include_raw=True
        )
        self.prompt = create_prompt_template(KEY_POINTS_SYSTEM_PROMPT)
        self.chain = self.prompt | self.structured_model

    def extract(self, development_report: str) -> Dict[str, str]:
        """
        Извлекает ключевые параметры из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Словарь с ключами: target_nature, business_task, ml_task,
            model_inference_scenario, model_application_segment.
            При ошибке каждое поле содержит сообщение об ошибке.
        """
        log_llm_request(
            logger, "KeyPointsExtractor", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = self.chain.invoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "KeyPointsExtractor", _raw, _elapsed)
            result: KeyPointsOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return {
                'Природа целевой переменной': result.target_nature,
                'Бизнес задача': result.business_task,
                'ML-задача': result.ml_task,
                'Сценарий использования модели': result.model_inference_scenario,
                'Сегмент данных модели': result.model_application_segment,
            }
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "KeyPointsExtractor", e, _elapsed)
            _err = f"Ошибка при извлечении: {e}"
            return {
                'Природа целевой переменной': _err,
                'Бизнес задача': _err,
                'ML-задача': _err,
                'Сценарий использования модели': _err,
                'Сегмент данных модели': _err,
            }

    async def aextract(self, development_report: str) -> Dict[str, str]:
        """
        Асинхронно извлекает ключевые параметры из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Словарь с ключами: target_nature, business_task, ml_task,
            model_inference_scenario, model_application_segment.
        """
        log_llm_request(
            logger, "KeyPointsExtractor (async)", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = await self.chain.ainvoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "KeyPointsExtractor (async)", _raw, _elapsed)
            result: KeyPointsOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return {
                'Природа целевой переменной': result.target_nature,
                'Бизнес задача': result.business_task,
                'ML-задача': result.ml_task,
                'Сценарий использования модели': result.model_inference_scenario,
                'Сегмент данных модели': result.model_application_segment,
            }
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "KeyPointsExtractor (async)", e, _elapsed)
            _err = f"Ошибка при извлечении: {e}"
            return {
                'Природа целевой переменной': _err,
                'Бизнес задача': _err,
                'ML-задача': _err,
                'Сценарий использования модели': _err,
                'Сегмент данных модели': _err,
            }
