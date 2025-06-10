import io
import re
from typing import List
from loguru import logger

# 导入所有我们需要用到的、具体的解析函数
from unstructured.partition.text import partition_text
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.md import partition_md
from unstructured.documents.elements import Element

def _normalize_whitespace(text: str) -> str:
    """
    规范化文本中的空白字符，合并多余的换行和空格。
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
    一个统一的函数，通过 match case 手动路由到正确的解析器。
    """
    logger.info(f"开始手动路由解析文件: {original_filename} (Content-Type: {content_type})...")
    
    elements: List[Element] = []
    
    try:
        # --- 核心改动：使用 match case 根据 content_type 选择解析器 ---
        match content_type:
            case "text/plain":
                logger.info("匹配到 text/plain，使用 partition_text 解析...")
                # partition_text 需要字符串，所以我们要先解码
                elements = partition_text(text=file_content_bytes.decode('utf-8', errors='ignore'))
            
            case "application/pdf":
                logger.info("匹配到 PDF，使用 partition_pdf 解析...")
                elements = partition_pdf(file=io.BytesIO(file_content_bytes), strategy="fast")

            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                logger.info("匹配到 DOCX，使用 partition_docx 解析...")
                elements = partition_docx(file=io.BytesIO(file_content_bytes))

            case "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                logger.info("匹配到 PPTX，使用 partition_pptx 解析...")
                elements = partition_pptx(file=io.BytesIO(file_content_bytes))
            
            case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                logger.info("匹配到 XLSX，使用 partition_xlsx 解析...")
                elements = partition_xlsx(file=io.BytesIO(file_content_bytes))

            case "text/markdown":
                logger.info("匹配到 Markdown，使用 partition_md 解析...")
                elements = partition_md(text=file_content_bytes.decode('utf-8', errors='ignore'))
            
            case _:
                # 对于未明确处理的类型，抛出错误或记录警告
                unsupported_message = f"不支持的内容类型: {content_type} (文件名: {original_filename})"
                logger.error(unsupported_message)
                raise ValueError(unsupported_message)

        if not elements:
            logger.warning(f"Unstructured 未能从文件 {original_filename} 中解析出任何元素。")
            return ""

        raw_text = "\n\n".join([str(el) for el in elements])
        logger.info(f"解析完成，提取原始文本长度: {len(raw_text)}")
        
        cleaned_text = _normalize_whitespace(raw_text)
        logger.info(f"文本规范化完成，最终文本长度: {len(cleaned_text)}")
        
        return cleaned_text

    except Exception as e:
        logger.error(f"使用 {content_type} 解析器处理文件 {original_filename} 时失败: {e}", exc_info=True)
        raise ValueError(f"文档解析失败: {original_filename}")