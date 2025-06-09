import io
import re
from typing import List
from loguru import logger
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element

def _normalize_whitespace(text: str) -> str:
    """
    规范化文本中的空白字符，我们之前讨论过的函数。
    """
    if not isinstance(text, str):
        return ""
    text = text.replace('\u200b', '')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text

def parse_and_clean_document(
    file_content_bytes: bytes,
    original_filename: str,
    content_type: str
) -> str:
    """
    一个统一的函数，用于解析任何支持的文档类型并进行清洗。    

    :param file_content_bytes: 从 MinIO 下载的文件原始字节流。
    :param original_filename: 原始文件名，用于 unstructured 推断文件类型。
    :return: 清洗和规范化后的纯文本内容。
    """
    logger.info(f"开始使用 unstructured 解析文件: {original_filename}...")
    
    try:
        # 1. 使用 unstructured 进行智能解析
        # partition 函数能自动识别文件类型（基于文件名后缀）并选择合适的解析器
        # 我们需要将 bytes 包装成一个文件类的对象 io.BytesIO
        elements: List[Element] = partition(
            file=io.BytesIO(file_content_bytes),
            file_filename=original_filename,
            content_type=content_type,
            strategy="fast"
            # 可以根据需要更换其他参数，例如 strategy="hi_res" 以获得更高质量的PDF解析
        )
        
        # 将解析出的所有元素拼接成一个长文本
        raw_text = "\n\n".join([str(el) for el in elements])
        logger.info(f"Unstructured 解析完成，提取原始文本长度: {len(raw_text)}")

        # 2. 进行二次清洗和规范化
        cleaned_text = _normalize_whitespace(raw_text)
        logger.info(f"文本规范化完成，最终文本长度: {len(cleaned_text)}")
        
        return cleaned_text

    except Exception as e:
        logger.error(f"使用 unstructured 解析文件 {original_filename} 时失败: {e}", exc_info=True)
        # 抛出一个 ValueError，以便上层逻辑可以捕获并处理
        raise ValueError(f"文档解析失败: {original_filename}")