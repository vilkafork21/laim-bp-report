import html as _html
import re
from typing import Dict, Any, List

import pandas as pd
from jinja2 import Template


class BusinessProcessVisualizer:
    """
    Visualizer for short business process descriptions.
    
    This class generates HTML reports that include:
    1. A plain text summary section (Краткое описание бизнес процесса) describing the model and business process
    2. A DataFrame table (Ключевые параметры) with key points about the business process
    
    The visualizer uses the same design and styling as the test results visualizer,
    with Arial font family and consistent table styling.
    """
    
    def __init__(self):
        self.html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 5px;
                    word-break: normal;
                    overflow-wrap: break-word;
                    hyphens: none;
                    text-align: left;
                }
                
                .description { 
                    margin: 5px 0; 
                    padding: 8px; 
                    background-color: #f0f8fa; 
                    border-radius: 5px;
                    border: 1px solid #d0e0ff;
                    line-height: 1;
                    white-space: pre-wrap;
                    display: inline-block;
                    min-width: 80%;
                    box-sizing: border-box;
                    text-align: left;
                }
                
                .description h3 {
                    font-size: 1.8em !important;
                    font-weight: bold !important;
                    margin: 0 0 15px 0 !important;
                    color: #333 !important;
                    text-align: left;
                }
                
                .key-points-header {
                    font-size: 1.8em !important;
                    font-weight: bold !important;
                    margin: 20px 0 10px 0 !important;
                    color: #333 !important;
                    text-align: left;
                }
                
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 10px 0;
                }
                
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    word-break: normal;
                    overflow-wrap: break-word;
                    hyphens: none;
                }
                
                th { 
                    background-color: #f2f2f2; 
                    font-size: 1.3em !important;
                    font-weight: bold !important;
                }
                
                tr:nth-child(even) { 
                    background-color: #f9f9f9; 
                }
                
                h2 { 
                    margin: 0; 
                    color: #333;
                    text-align: left; 
                }
                
                h3 { 
                    margin: 5px 0; 
                    color: #333; 
                    font-size: 1.4em;
                    font-weight: bold;
                    text-align: left;
                }
                
                h4 { 
                    margin: 5px 0 10px 0; 
                    color: #333; 
                    text-align: left;
                }
            </style>
        </head>
        <body>
            {% if summary %}
            <h3>Краткое описание бизнес процесса</h3>
            <div class="description">
                <div>{{ summary }}</div>
            </div>
            {% endif %}
            
            {% if key_points %}
            <div>
                <h3 class="key-points-header">Ключевые параметры</h3>
                {{ key_points|safe }}
            </div>
            {% endif %}

            {% if generation_hyperparams %}
            <div>
                <h3 class="key-points-header">Гиперпараметры генерации</h3>
                {{ generation_hyperparams|safe }}
            </div>
            {% endif %}

            {% if sample_description %}
            <div>
                <h3 class="key-points-header">Описание выборок</h3>
                {{ sample_description|safe }}
            </div>
            {% endif %}

            {% if ml_architecture %}
            <div>
                <h3 class="key-points-header">Подробности архитектуры ML решения</h3>
                <div class="description">
                    <div>{{ ml_architecture }}</div>
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """)
    
    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """Экранирует HTML-спецсимволы и заменяет переносы строк на <br>."""
        text = _html.escape(text)
        text = text.strip('\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace('\n', '<br>')
        return text

    def visualize(self,
                  title: str = "Описание бизнес-процесса",
                  summary: str = None,
                  key_points: Dict[str, str] = None,
                  generation_hyperparams: Dict[str, Any] = None,
                  sample_description: List[Dict[str, str]] = None,
                  ml_architecture: str = None) -> str:
        """
        Генерирует HTML-визуализацию описания бизнес-процесса.

        Весь LLM-контент HTML-экранируется перед вставкой в отчёт.

        Args:
            title: Заголовок отчёта. Рендерится в теге <title> и используется
                   как идентификатор документа. По умолчанию "Описание бизнес-процесса".
            summary: Суммаризация — свободный текст, описывающий модель и бизнес-процесс.
                     Результат SummaryGenerator.generate().
            key_points: Словарь ключевых параметров бизнес-процесса.
                        Результат KeyPointsExtractor.extract().
                        Ожидаемые ключи (русскоязычные):
                        'Природа целевой переменной', 'Бизнес задача', 'ML-задача',
                        'Сценарий использования модели', 'Сегмент данных модели'.
                        Рендерится как двухколоночная таблица ('Параметр', 'Описание').
            generation_hyperparams: Словарь гиперпараметров генерации.
                        Результат HyperparamsExtractor.extract().
                        Рендерится как двухколоночная таблица ('Гиперпараметр', 'Значение').
            sample_description: Список словарей с описанием выборок.
                        Результат SampleExtractor.extract().
                        Каждый словарь должен содержать ключи:
                        - 'split'  (train / val / oos / oot)
                        - 'thema'  (тематика / канал / '-' если однородно)
                        - 'volume' (количество наблюдений int или None)
                        - 'dates'  (период сбора или '-')
                        Рендерится как четырёхколоночная таблица
                        ('Часть выборки', 'Тематика', 'Объем выборки', 'Даты сбора').
            ml_architecture: Свободный текст с описанием архитектуры ML-решения.
                        Результат MlArchitectureExtractor.extract().

        Returns:
            str: HTML-код отчёта.

        Example:
            >>> visualizer = BusinessProcessVisualizer()
            >>> summary = "This model predicts customer churn in the telecommunications industry..."
            >>> key_points = {
            ...     'target_nature': 'Binary classification (churn vs no churn)',
            ...     'business_task': 'Customer retention and loyalty management',
            ...     'ml_task': 'Supervised learning - classification',
            ...     'model_inference_scenario': 'Batch processing for monthly customer analysis',
            ...     'model_application_segment': 'Active subscribers with >= 3 months of history'
            ... }
            >>> generation_hyperparams = {
            ...     'temperature': 0.7,
            ...     'max_tokens': 1024,
            ...     'top_p': 0.95
            ... }
            >>> sample_description = [
            ...     {'split': 'train', 'thema': 'звонки в КЦ', 'volume': 50000, 'dates': '2022–2023'},
            ...     {'split': 'train', 'thema': 'чат-сессии',  'volume': 12000, 'dates': '2022–2023'},
            ...     {'split': 'val',   'thema': '-',            'volume': 10000, 'dates': '2024 Q1'},
            ...     {'split': 'oot',   'thema': '-',            'volume': None,  'dates': '2024 Q2'},
            ... ]
            >>> html = visualizer.visualize(
            ...     title="Customer Churn Prediction",
            ...     summary=summary,
            ...     key_points=key_points,
            ...     generation_hyperparams=generation_hyperparams,
            ...     sample_description=sample_description
            ... )
        """

        if summary:
            summary = self._normalize_newlines(summary)

        if ml_architecture:
            ml_architecture = self._normalize_newlines(ml_architecture)

        # Convert key points dictionary to DataFrame HTML
        key_points_html = None
        if key_points:
            # Create DataFrame from dictionary with Russian column names
            df = pd.DataFrame(list(key_points.items()),
                            columns=['Параметр', 'Описание'])
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: self._normalize_newlines(x) if isinstance(x, str) else x
                )
            # Convert to HTML with styling
            key_points_html = df.to_html(index=False,
                                       classes='dataframe',
                                       escape=False,
                                       border=0)

        # Convert generation hyperparameters dictionary to DataFrame HTML
        generation_hyperparams_html = None
        if generation_hyperparams:
            df = pd.DataFrame(list(generation_hyperparams.items()),
                              columns=['Гиперпараметр', 'Значение'])
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: self._normalize_newlines(x) if isinstance(x, str) else x
                )
            generation_hyperparams_html = df.to_html(index=False,
                                                     classes='dataframe',
                                                     escape=False,
                                                     border=0)

        # Convert sample description list of dicts to DataFrame HTML (4 columns)
        sample_description_html = None
        if sample_description:
            df = pd.DataFrame(
                [
                    {
                        'Часть выборки': s['split'],
                        'Тематика': s['thema'],
                        'Объем выборки': s['volume'] if s['volume'] is not None else '-',
                        'Даты сбора': s['dates'],
                    }
                    for s in sample_description
                ]
            )
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: self._normalize_newlines(x) if isinstance(x, str) else x
                )
            sample_description_html = df.to_html(index=False,
                                                 classes='dataframe',
                                                 escape=False,
                                                 border=0)

        # Render template
        html = self.html_template.render(
            title=title,
            summary=summary,
            key_points=key_points_html,
            generation_hyperparams=generation_hyperparams_html,
            sample_description=sample_description_html,
            ml_architecture=ml_architecture,
        )
        
        return html
    
    @staticmethod
    def save_to_file(html: str, filepath: str) -> None:
        """
        Save the generated HTML to a file.
        
        Args:
            html: HTML content to save
            filepath: Path where the HTML file should be saved
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
