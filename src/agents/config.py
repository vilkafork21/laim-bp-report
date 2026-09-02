"""
Конфигурация GigaChat для агентов извлечения бизнес-процесса.
Использует словарь generation_config, который распаковывается при инициализации GigaChat.
"""

from typing import Optional
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate


DEFAULT_GENERATION_CONFIG: dict = {
    'model': 'GigaChat-Pro',
    'temperature': 0.0,
    'max_tokens': 2048,
}


def get_chat_model(generation_config: dict) -> GigaChat:
    """
    Создает экземпляр GigaChat.

    Args:
        credentials: API-ключ GigaChat
        generation_config: Словарь с гиперпараметрами генерации.
                           Поддерживаемые ключи: model, temperature, max_tokens,
                           top_p, top_k, repetition_penalty и др.
                           Распаковывается напрямую в конструктор GigaChat.

    Returns:
        Экземпляр GigaChat
    """
    return GigaChat(**generation_config)


def create_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """
    Создает шаблон промпта с системным сообщением.

    Args:
        system_prompt: Системный промпт

    Returns:
        ChatPromptTemplate с переменными {input}
    """
    return ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', '{input}'),
    ])


def create_two_var_prompt_template(system_prompt: str, human_template: str) -> ChatPromptTemplate:
    """
    Создает шаблон промпта с произвольным human-сообщением (несколько переменных).

    Args:
        system_prompt: Системный промпт
        human_template: Шаблон человеческого сообщения с переменными в формате {var}

    Returns:
        ChatPromptTemplate
    """
    return ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', human_template),
    ])
