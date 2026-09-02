"""
Pipeline 4. Извлечение описания выборок (датасетов) из отчёта о разработке.

Для каждой записи извлекает:
  - split:  часть выборки (train / val / oos / oot)
  - thema:  тематика запросов / канал / направление / подзадача
  - volume: количество наблюдений (int) или None, если не указано
  - dates:  период сбора данных

Один split может давать несколько строк при разбиении по тематике/каналу.
"""

import time
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from ..config import DEFAULT_GENERATION_CONFIG, get_chat_model, create_prompt_template
from ..prompts import SAMPLE_SYSTEM_PROMPT
from src.utils.llm_logging import log_llm_request, log_llm_response, log_llm_error

logger = logging.getLogger(__name__)


class SampleRecord(BaseModel):
    """Описание одной записи выборки (пара split × thema)."""

    split: str = Field(description="Часть выборки: train, val, oos или oot.")
    thema: str = Field(
        description=(
            "Тематика запросов / канал / направление / подзадача. "
            "'-' если разбиения нет и данные однородны."
        )
    )
    volume: Optional[int] = Field(
        default=None,
        description=(
            "Количество наблюдений в виде целого числа. "
            "null если объём явно не указан в отчёте."
        ),
    )
    dates: str = Field(description="Период сбора данных. '-' если не указан.")


class SampleDescriptionOutput(BaseModel):
    """Список записей выборок, извлечённых из отчёта."""

    samples: List[SampleRecord] = Field(
        description=(
            "Список записей. Один split может давать несколько строк "
            "при разбиении по тематике/каналу. "
            "Пустой список, если информация о выборках отсутствует."
        )
    )


class SampleExtractor:
    """
    Pipeline 4. Извлекает описание выборок из отчёта о разработке.

    Возвращает список словарей с ключами split, thema, volume, dates — пригодный
    для передачи в BusinessProcessVisualizer.visualize() в качестве аргумента
    sample_description.

    Example:
        >>> extractor = SampleExtractor(
        ...     generation_config={'model': 'GigaChat-2-Max', 'temperature': 0.0, 'max_tokens': 2048}
        ... )
        >>> samples = extractor.extract(development_report)
        >>> # [
        >>> #   {'split': 'train', 'thema': 'звонки в КЦ',  'volume': 50000, 'dates': '2022–2023'},
        >>> #   {'split': 'train', 'thema': 'чат-сессии',   'volume': 12000, 'dates': '2022–2023'},
        >>> #   {'split': 'val',   'thema': '-',             'volume': 10000, 'dates': '2024 Q1'},
        >>> #   {'split': 'oot',   'thema': '-',             'volume': None,  'dates': '2024 Q2'},
        >>> # ]
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
        self.structured_model = chat_model.with_structured_output(
            SampleDescriptionOutput, include_raw=True
        )
        self.prompt = create_prompt_template(SAMPLE_SYSTEM_PROMPT)
        self.chain = self.prompt | self.structured_model

    @staticmethod
    def _record_to_dict(s: SampleRecord) -> Dict[str, Any]:
        return {
            'split': s.split,
            'thema': s.thema,
            'volume': s.volume,   # int или None
            'dates': s.dates,
        }

    def extract(self, development_report: str) -> List[Dict[str, Any]]:
        """
        Извлекает описание выборок из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Список словарей с ключами:
              - 'split':  часть выборки (train / val / oos / oot)
              - 'thema':  тематика / канал / направление ('-' если однородно)
              - 'volume': количество наблюдений (int) или None
              - 'dates':  период сбора данных
            Пустой список, если информация о выборках в отчёте отсутствует.
            При ошибке возвращает список с одним словарём-заглушкой.
        """
        log_llm_request(
            logger, "SampleExtractor", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = self.chain.invoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "SampleExtractor", _raw, _elapsed)
            result: SampleDescriptionOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return [self._record_to_dict(s) for s in result.samples]
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "SampleExtractor", e, _elapsed)
            return [{'split': 'error', 'thema': '-', 'volume': None, 'dates': f"Ошибка: {e}"}]

    async def aextract(self, development_report: str) -> List[Dict[str, Any]]:
        """
        Асинхронно извлекает описание выборок из отчёта о разработке.

        Args:
            development_report: Текст отчёта о разработке модели.

        Returns:
            Список словарей с ключами split, thema, volume, dates.
        """
        log_llm_request(
            logger, "SampleExtractor (async)", development_report,
            self.generation_config.get('model', ''),
            self.generation_config.get('temperature', 0.0),
            self.generation_config.get('max_tokens', 0),
        )
        _start = time.time()
        try:
            _raw = await self.chain.ainvoke(development_report)
            _elapsed = time.time() - _start
            log_llm_response(logger, "SampleExtractor (async)", _raw, _elapsed)
            result: SampleDescriptionOutput = (
                _raw.get('parsed') if isinstance(_raw, dict) else _raw
            )
            return [self._record_to_dict(s) for s in result.samples]
        except Exception as e:
            _elapsed = time.time() - _start
            log_llm_error(logger, "SampleExtractor (async)", e, _elapsed)
            return [{'split': 'error', 'thema': '-', 'volume': None, 'dates': f"Ошибка: {e}"}]
