# g-aiva-business-process

Модуль извлечения бизнес-процессного описания из отчёта о разработке ML-модели. Генерирует HTML-отчёт с пятью блоками извлечённой информации.

## Что делает модуль

Принимает отчёт о разработке модели, прогоняет его через 5 LLM-пайплайнов на GigaChat и возвращает HTML-отчёт.

| Pipeline | Экстрактор | Результат |
|---|---|---|
| 1 | `KeyPointsExtractor` | Ключевые параметры: бизнес-задача, ML-задача, сценарий применения, сегмент |
| 2 | `SummaryGenerator` | Краткое описание: проблема → модель → решение |
| 3 | `HyperparamsExtractor` | Гиперпараметры генерации LLM из отчёта |
| 4 | `SampleExtractor` | Описание выборок (split / тематика / объём / даты) |
| 5 | `MlArchitectureExtractor` | Архитектура ML-решения от входных до выходных данных |

## Структура

```
main.py                          # точка входа
descriptor.json                  # дескриптор сервиса
src/
  agents/
    business_process_agent.py    # агрегатор пайплайнов
    config.py                    # конфигурация GigaChat
    configs/
      default_business_process_config.json
    extractors/                  # 5 LLM-экстракторов
    prompts/                     # системные промпты
  file_parsing/                  # парсеры docx / pdf / xlsx
  utils/
    llm_logging.py               # утилиты логирования LLM-вызовов
  visualizer.py                  # генератор HTML-отчёта
```

## Входные / выходные данные

**Вход** (`development_report`): бинарный файл отчёта о разработке в формате `{"bin": bytes, "ext": str}`. Поддерживаемые форматы: `.docx`, `.doc`, `.pdf`, `.xlsx`, `.xls`.

**Выход** (`hidden_port`): HTML-строка с отчётом.

## Параметры настройки

Все параметры LLM задаются через единый блок `llm parameters` в UI или как kwargs при вызове:

| Параметр | Тип | По умолчанию |
|---|---|---|
| `pr_llm_model` | string | `GigaChat-2-Max` |
| `pr_llm_temperature` | float | `0.0` |
| `pr_llm_top_p` | float | `0.1` |
| `pr_llm_repetition_penalty` | float | `1` |
| `pr_llm_max_tokens` | int | `2048` |

Для изменения дефолтных значений отредактируйте `src/agents/configs/default_business_process_config.json`.
