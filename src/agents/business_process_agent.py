"""
Агрегатор бизнес-процессного анализа отчёта о разработке модели.

Прогоняет отчёт через все 5 LLM-пайплайнов и возвращает словарь kwargs,
готовый к передаче напрямую в BusinessProcessVisualizer.visualize().

"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractors import (
    HyperparamsExtractor,
    KeyPointsExtractor,
    MlArchitectureExtractor,
    SampleExtractor,
    SummaryGenerator,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY_SEC = 2.0


def _retry(fn, *args, max_retries: int = _MAX_RETRIES, delay: float = _RETRY_DELAY_SEC, **kwargs):
    """
    Запускает fn(*args, **kwargs) с повторными попытками при исключении.

    Args:
        fn: Вызываемый объект.
        *args: Позиционные аргументы для fn.
        max_retries: Максимальное число попыток (включая первую).
        delay: Пауза в секундах между попытками.
        **kwargs: Именованные аргументы для fn.

    Returns:
        Результат fn при успехе.

    Raises:
        Exception: Последнее исключение, если все попытки исчерпаны.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[_retry] attempt %d/%d failed: %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exc


def business_process_agent(
    development_report: str,
    llm_config: Optional[Dict[str, Any]] = None,
    llm_config_path: Optional[str] = None,
    sds_llm_config: Optional[Dict[str, Any]] = None,
    title: str = "Описание бизнес-процесса",
) -> Dict[str, Any]:
    """
    Запускает 5 LLM-пайплайнов над отчётом о разработке и возвращает kwargs
    для BusinessProcessVisualizer.visualize().

    Каждый пайплайн выполняется с автоматическим повтором до 5 раз при ошибке.

    Порядок выполнения:
      1. KeyPointsExtractor       — ключевые параметры
      2. SummaryGenerator         — суммаризация (зависит от результата п.1)
      3. HyperparamsExtractor     — гиперпараметры генерации
      4. SampleExtractor          — описание выборок
      5. MlArchitectureExtractor  — архитектура ML-решения

    Args:
        development_report: Текст отчёта о разработке модели.
        llm_config: Опциональный словарь конфигурации LLM. Если None — загружается
                    через load_config(llm_config_path). Ожидается структура
                    {'llm': {'kwargs': {...}}}.
        llm_config_path: Опциональный путь до json-конфига. Если None — используется
                         дефолтный конфиг из configs/default_business_process_config.json.
        sds_llm_config: Словарь дополнительных параметров подключения к LLM-шлюзу
                    (например, base_url, verify_ssl_certs, timeout).
                    Мержится поверх llm_config['llm']['kwargs'].
                    Если None — дополнительные параметры не добавляются.
        title: Заголовок HTML-отчёта. Передаётся в visualize() и рендерится
               в теге <title> и заголовке страницы.

    Returns:
        Словарь kwargs, совместимый с BusinessProcessVisualizer.visualize():
        {
            'title': str,
            'summary': str,
            'key_points': Dict[str, str],
            'generation_hyperparams': Dict[str, str],
            'sample_description': List[Dict[str, str]],
            'ml_architecture': str,
        }
    """
    if llm_config is None:
        config = load_config(llm_config_path)
    else:
        config = llm_config

    _generation_config = config['llm']['kwargs']
    if sds_llm_config:
        _generation_config = _generation_config | sds_llm_config

    logger.info("[business_process_agent] start | report_len=%d", len(development_report))

    kp_extractor = KeyPointsExtractor(generation_config=_generation_config)
    sg_generator = SummaryGenerator(generation_config=_generation_config)
    hp_extractor = HyperparamsExtractor(generation_config=_generation_config)
    sa_extractor = SampleExtractor(generation_config=_generation_config)
    ml_extractor = MlArchitectureExtractor(generation_config=_generation_config)

    # Pipeline 1: ключевые параметры
    logger.info("[business_process_agent] pipeline 1: KeyPointsExtractor")
    key_points: Dict[str, str] = _retry(kp_extractor.extract, development_report)

    # Pipeline 2: суммаризация (зависит от key_points)
    logger.info("[business_process_agent] pipeline 2: SummaryGenerator")
    summary: str = _retry(sg_generator.generate, development_report, key_points)

    # Pipeline 3: гиперпараметры генерации
    logger.info("[business_process_agent] pipeline 3: HyperparamsExtractor")
    generation_hyperparams: Dict[str, str] = _retry(hp_extractor.extract, development_report)

    # Pipeline 4: описание выборок
    logger.info("[business_process_agent] pipeline 4: SampleExtractor")
    sample_description: List[Dict[str, str]] = _retry(sa_extractor.extract, development_report)

    # Pipeline 5: архитектура ML-решения
    logger.info("[business_process_agent] pipeline 5: MlArchitectureExtractor")
    ml_architecture: str = _retry(ml_extractor.extract, development_report)

    logger.info("[business_process_agent] done")

    return {
        'title': title,
        'summary': summary,
        'key_points': key_points,
        'generation_hyperparams': generation_hyperparams,
        'sample_description': sample_description,
        'ml_architecture': ml_architecture,
    }


def load_config(path: Optional[str] = None) -> dict:
    """
    Загружает json-конфиг по пути с подстановкой дефолтного варианта.

    Args:
        path: Путь до конфига

    Returns:
        dict-конфиг
    """
    if path is not None:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        base_dir = Path(__file__).resolve().parent
        default_config_path = base_dir / "configs" / "default_business_process_config.json"
        with open(default_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
