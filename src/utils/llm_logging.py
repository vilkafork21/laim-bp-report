"""
Утилиты для логирования вызовов LLM и извлечений.

Предоставляет функции для логирования запросов и ответов LLM,
ошибок и хода выполнения пайплайнов извлечения.
Все функции только логируют и не зависят от вызовов в модель.
"""

import logging
from typing import Any


def log_llm_request(
    logger: logging.Logger,
    chain_name: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> None:
    """
    Логирует детали запроса перед вызовом LLM.

    Args:
        logger: Логгер модуля
        chain_name: Название цепочки/извлечения для идентификации в логах
        context: Входной текст запроса
        model: Название модели
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов
    """
    logger.info(
        f"[{chain_name}] LLM вызов | "
        f"model={model} | "
        f"temperature={temperature} | "
        f"max_tokens={max_tokens}"
    )
    logger.info(
        f"[{chain_name}] Входной запрос | длина={len(context)} символов"
    )
    logger.debug(
        f"[{chain_name}] Входной запрос (текст): {context[:500]}"
    )


def log_llm_response(
    logger: logging.Logger,
    chain_name: str,
    raw_result: Any,
    elapsed: float
) -> None:
    """
    Логирует результат вызова LLM после его завершения.

    Обрабатывает оба формата результата:
    - dict с ключами 'raw'/'parsed' (from include_raw=True)
    - plain AIMessage

    Проверяет finish_reason и выводит отдельное предупреждение при значении 'blacklist'.

    Args:
        logger: Логгер модуля
        chain_name: Название цепочки/извлечения для идентификации в логах
        raw_result: Сырой результат вызова цепочки
        elapsed: Время выполнения запроса в секундах
    """
    raw_msg = raw_result.get("raw") if isinstance(raw_result, dict) else raw_result
    if raw_msg is None:
        logger.info(f"[{chain_name}] Завершено за {elapsed:.2f}с (метаданные недоступны)")
        return

    metadata = getattr(raw_msg, 'response_metadata', {}) or {}
    finish_reason = metadata.get('finish_reason', 'unknown')
    token_usage = metadata.get('token_usage', {}) or {}
    input_tokens = token_usage.get('prompt_tokens', 'n/a')
    output_tokens = token_usage.get('completion_tokens', 'n/a')

    out_text = getattr(raw_msg, 'content', '')
    logger.info(
        f"[{chain_name}] Выходной ответ | длина={len(str(out_text))} символов"
    )
    logger.debug(
        f"[{chain_name}] Выходной ответ (текст): {str(out_text)[:500]}"
    )
    logger.info(
        f"[{chain_name}] Завершено | "
        f"finish_reason={finish_reason} | "
        f"input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | "
        f"время={elapsed:.2f}с"
    )

    if finish_reason == 'blacklist':
        logger.warning(
            f"[{chain_name}] Запрос заблокирован blacklist-фильтром GigaChat!"
        )


def log_llm_error(
    logger: logging.Logger,
    chain_name: str,
    error: Exception,
    elapsed: float
) -> None:
    """
    Логирует ошибку при вызове LLM.

    Args:
        logger: Логгер модуля
        chain_name: Название цепочки/извлечения для идентификации в логах
        error: Исключение, произошедшее при вызове
        elapsed: Время до возникновения ошибки в секундах
    """
    logger.info(
        f"[{chain_name}] Ошибка при вызове LLM за {elapsed:.2f}с: {error}"
    )
