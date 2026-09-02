import pdfplumber
import pandas as pd
from typing import List, Dict, Any

def is_char_inside_bboxes(char: Dict[str, Any], bboxes: List[tuple]) -> bool:
    """
    Проверяет, находится ли центр символа внутри одного из прямоугольников (bboxes).
    Parameters:
        char: Словарь с информацией о символе от pdfplumber.
        bboxes: Список кортежей, где каждый кортеж - это bbox (x0, top, x1, bottom).
    Returns:
        True, если символ внутри хотя бы одного bbox, иначе False.
    """
    v_center = (char['top'] + char['bottom']) / 2
    h_center = (char['x0'] + char['x1']) / 2
    
    for bbox in bboxes:
        x0, top, x1, bottom = bbox
        if (h_center >= x0 and h_center <= x1 and 
            v_center >= top and v_center <= bottom):
            return True
    return False

def pdf_table_to_md(table: List[List[str]]):
    df = pd.DataFrame(table[1:], columns=table[0])
    return df.to_markdown()

def parse_pdf(pdf_path: str) -> str:
    pdf2md = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Извлекаем все таблицы на текущей странице
            # table_settings можно настроить для лучшего распознавания
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            all_chars = page.chars
                
            # 3. Фильтруем символы, оставляя только те, что НЕ находятся внутри таблиц
            chars_outside_tables = [
                char for char in all_chars 
                if not is_char_inside_bboxes(char, table_bboxes)
            ]

            text_on_page = pdfplumber.utils.extract_text(chars_outside_tables)
            tables_on_page = page.extract_tables()

            for table in tables_on_page:
                pdf2md.append(pdf_table_to_md(table) + "\n\n")
                
            pdf2md.append(text_on_page + "\n\n") # Добавляем разделитель между страницами

    return "".join(pdf2md)
