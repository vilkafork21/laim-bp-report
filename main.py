import io
import json
import logging
from typing import Dict, Any

from src.file_parsing.parser import parse_file
from src.visualizer import BusinessProcessVisualizer
from src.agents.business_process_agent import business_process_agent, load_config

logger = logging.getLogger(__name__)

SDS_LLM_CONFIG = {
    "base_url": "http://sds-ai-gateway:8097/api/v1",
    "verify_ssl_certs": False,
    "profanity_check": True,
    "timeout": 500,
}


def get_report_from_path(binary_data: io.BytesIO, file_ext: str):
    file_ext = file_ext.lower()
    parsed = parse_file(binary_data, file_ext)
    return parsed


def main(
    development_report: Dict[str, Any],
    **kwargs) -> Dict[str, Any]:

    binary_report = io.BytesIO(development_report["bin"])
    report_extension = development_report["ext"]
    report = get_report_from_path(binary_report, report_extension)
    report_text = report.get_all_plain_text()

    llm_config = load_config()
    llm_config['llm']['kwargs']['model'] = kwargs.get('pr_llm_model', llm_config['llm']['kwargs']['model'])
    llm_config['llm']['kwargs']['temperature'] = kwargs.get('pr_llm_temperature', llm_config['llm']['kwargs']['temperature'])
    llm_config['llm']['kwargs']['top_p'] = kwargs.get('pr_llm_top_p', llm_config['llm']['kwargs'].get('top_p'))
    llm_config['llm']['kwargs']['repetition_penalty'] = kwargs.get('pr_llm_repetition_penalty', llm_config['llm']['kwargs'].get('repetition_penalty'))
    llm_config['llm']['kwargs']['max_tokens'] = kwargs.get('pr_llm_max_tokens', llm_config['llm']['kwargs']['max_tokens'])

    logger.info(f'Стартуем извлечение со следующими параметрами LLM: \n {json.dumps(llm_config, indent=4, ensure_ascii=False)}')

    vis_kwargs = business_process_agent(
        development_report=report_text,
        llm_config=llm_config,
        sds_llm_config=SDS_LLM_CONFIG,
    )

    logger.info('Извлечение завершено, формируем HTML-отчёт.')

    html_report = BusinessProcessVisualizer().visualize(**vis_kwargs)

    return {
        "hidden_port": html_report,
    }
