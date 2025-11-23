from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

try:
    import mammoth
    HAS_MAMMOTH = True
except ImportError:
    HAS_MAMMOTH = False

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


def _iter_block_items(parent: DocumentType | _Cell) -> Iterable[Paragraph | Table]:
    """
    Итерирует блоки документа (параграфы и таблицы) сохраняя порядок.
    """
    if isinstance(parent, DocumentType):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def _normalize_text(text: str) -> str:
    """
    Удаляет лишние переводы строк/пробелы внутри блока.
    """
    clean = re.sub(r"\s+", " ", text.replace("\r", " ")).strip()
    return clean


def _is_numbered_list(document: DocumentType, numId: str, ilvl: int) -> bool:
    """
    Определяет, является ли список нумерованным, проверяя numbering.xml документа.
    """
    try:
        # Пытаемся найти определение списка в numbering.xml
        numbering_part = document.part.numbering_part
        if numbering_part is None:
            return False
        
        numbering_xml = numbering_part.element
        # Ищем num с нужным numId
        for num in numbering_xml.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num"):
            num_id_attr = num.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numId")
            if num_id_attr == numId:
                # Нашли определение списка, проверяем формат
                abstract_num_id = num.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}abstractNumId")
                if abstract_num_id is not None:
                    abstract_num_id_val = abstract_num_id.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
                    # Ищем abstractNum
                    for abstract_num in numbering_xml.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}abstractNum"):
                        if abstract_num.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}abstractNumId") == abstract_num_id_val:
                            # Проверяем lvl для нужного уровня
                            for lvl in abstract_num.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}lvl"):
                                lvl_ilvl = lvl.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl")
                                if lvl_ilvl == str(ilvl):
                                    # Проверяем формат нумерации
                                    num_fmt = lvl.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numFmt")
                                    if num_fmt is not None:
                                        fmt_val = num_fmt.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
                                        # Если формат не "bullet" или "none", то это нумерованный список
                                        return fmt_val not in ["bullet", "none"]
    except Exception:
        # Если не удалось определить, используем fallback
        pass
    
    return False


def _get_list_prefix(paragraph: Paragraph, document: DocumentType | None = None) -> str | None:
    """
    Определяет префикс списка для параграфа (номер или маркер).
    Возвращает None, если параграф не является элементом списка.
    """
    pPr = paragraph._p.pPr
    if pPr is None:
        return None
    
    # Проверяем наличие нумерации
    numPr = pPr.numPr
    if numPr is None:
        return None
    
    # Получаем numId и ilvl
    numId_elem = numPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numId")
    ilvl_elem = numPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl")
    
    if numId_elem is None:
        return None
    
    numId = numId_elem.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
    ilvl = int(ilvl_elem.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "0")) if ilvl_elem is not None else 0
    
    # Используем стиль параграфа для определения типа списка
    style_name = paragraph.style.name.lower() if paragraph.style else ""
    
    # Определяем тип списка
    is_bullet = "bullet" in style_name
    is_number = "number" in style_name or ("list" in style_name and "number" in style_name)
    
    # Если стиль не помог и есть доступ к документу, проверяем numbering.xml
    if not is_bullet and not is_number and document is not None:
        is_number = _is_numbered_list(document, numId, ilvl)
    
    if is_bullet:
        # Маркированный список
        markers = ["•", "-", "◦", "▪"]
        marker = markers[min(ilvl, len(markers) - 1)]
        return marker + " "
    elif is_number:
        # Нумерованный список - возвращаем специальный маркер,
        # который будет заменен на номер при обработке
        return f"__LIST_NUM_{ilvl}__"
    else:
        # По умолчанию считаем маркированным
        markers = ["•", "-", "◦", "▪"]
        marker = markers[min(ilvl, len(markers) - 1)]
        return marker + " "


def _get_paragraph_text_with_breaks(paragraph: Paragraph) -> str:
    """
    Извлекает текст параграфа с сохранением разрывов строк внутри параграфа.
    """
    text_parts = []
    for run in paragraph.runs:
        run_text = run.text
        # Проверяем наличие разрывов строк в XML (w:br элементы)
        if run._element is not None:
            # Ищем все разрывы строк
            breaks = run._element.xpath('.//w:br')
            if breaks:
                # Если есть разрывы, добавляем текст и разрывы
                text_parts.append(run_text)
                text_parts.append('\n' * len(breaks))
            else:
                text_parts.append(run_text)
        else:
            text_parts.append(run_text)
    return ''.join(text_parts)


def _get_actual_list_number(paragraph: Paragraph, document: DocumentType, numbering_state: dict) -> str | None:
    """
    Извлекает реальный номер списка из Word документа.
    numbering_state - словарь для отслеживания состояния нумерации: {(numId, ilvl): counter}
    """
    pPr = paragraph._p.pPr
    if pPr is None:
        return None
    
    numPr = pPr.numPr
    if numPr is None:
        return None
    
    numId_elem = numPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numId")
    ilvl_elem = numPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl")
    
    if numId_elem is None:
        return None
    
    numId = numId_elem.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
    ilvl = int(ilvl_elem.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "0")) if ilvl_elem is not None else 0
    
    # Используем ключ (numId, ilvl) для отслеживания состояния нумерации
    key = (numId, ilvl)
    
    # Увеличиваем счетчик для этого списка и уровня
    if key not in numbering_state:
        numbering_state[key] = 0
    
    numbering_state[key] += 1
    
    # Сбрасываем счетчики для более глубоких уровней того же numId
    for (nid, level), counter in list(numbering_state.items()):
        if nid == numId and level > ilvl:
            del numbering_state[(nid, level)]
    
    # Формируем номер в зависимости от уровня
    if ilvl == 0:
        number = str(numbering_state[key])
    else:
        # Для вложенных списков формируем составной номер
        parts = []
        for level in range(ilvl + 1):
            level_key = (numId, level)
            if level_key in numbering_state:
                parts.append(str(numbering_state[level_key]))
            else:
                # Если счетчик для уровня не установлен, начинаем с 1
                parts.append("1")
        number = ".".join(parts)
    
    return number


def _format_paragraph_with_list(paragraph: Paragraph, numbering_state: dict, document: DocumentType | None = None) -> str:
    """
    Форматирует параграф с учетом списка.
    numbering_state - словарь для отслеживания состояния нумерации: {(numId, ilvl): counter}
    """
    # Используем функцию, которая сохраняет разрывы строк внутри параграфа
    text = _get_paragraph_text_with_breaks(paragraph).rstrip()
    
    # Проверяем, начинается ли текст с номера списка (например, "1.1. ", "2.1\t")
    # Если да, значит номер уже есть в тексте Word, и мы просто возвращаем текст как есть
    if re.match(r'^\d+(?:\.\d+)*[\.\)]\s+', text) or re.match(r'^\d+(?:\.\d+)*\t', text):
        # Номер уже есть в тексте - возвращаем как есть
        return text
    
    # Если номера нет в тексте, проверяем, является ли параграф элементом списка
    prefix = _get_list_prefix(paragraph, document)
    
    if prefix is None:
        # Не список - просто возвращаем текст
        return text
    
    if prefix.startswith("__LIST_NUM_"):
        # Нумерованный список - извлекаем реальный номер из Word документа
        if document is not None:
            actual_number = _get_actual_list_number(paragraph, document, numbering_state)
            if actual_number is not None:
                return f"{actual_number}. {text}"
        
        # Fallback: если не удалось извлечь реальный номер, используем старую логику
        ilvl = int(prefix.replace("__LIST_NUM_", "").replace("__", ""))
        # Используем старую логику как fallback
        if ilvl == 0:
            if (None, 0) not in numbering_state:
                numbering_state[(None, 0)] = 0
            numbering_state[(None, 0)] += 1
            number = str(numbering_state[(None, 0)])
        else:
            parts = []
            for level in range(ilvl + 1):
                key = (None, level)
                if key in numbering_state:
                    parts.append(str(numbering_state[key]))
                else:
                    parts.append("1")
            number = ".".join(parts)
        
        return f"{number}. {text}"
    else:
        # Маркированный список
        return f"{prefix}{text}"


def extract_text(document: DocumentType) -> str:
    """
    Возвращает текст документа, сохраняя исходные переносы строк и структуру списков.
    """
    lines: List[str] = []
    numbering_state: dict = {}  # Состояние нумерации: {(numId, ilvl): counter}
    
    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            # Проверяем, является ли параграф заголовком этапа (например, "Этап 2: Масштабирование")
            # или подраздела (например, "Подраздел 2.2 Стадийность (этапы)")
            # Если да, сбрасываем счетчики списков ПЕРЕД обработкой этого параграфа
            text = block.text.strip()
            # Проверяем различные паттерны заголовков, которые должны сбрасывать счетчики
            should_reset = (re.match(r'^Этап\s+\d+:', text, re.IGNORECASE) or
                           re.match(r'^Подраздел\s+\d+\.\d+', text, re.IGNORECASE) or
                           re.match(r'^РАЗДЕЛ\s+\d+\.', text, re.IGNORECASE))
            
            formatted = _format_paragraph_with_list(block, numbering_state, document)
            
            # Проверяем, является ли это пустым параграфом (не частью списка)
            if not formatted.strip():
                # Пустой параграф - добавляем пустую строку для сохранения структуры
                lines.append("")
            else:
                # Проверяем, что это не только номер списка без текста (например, "1. " или "2.1. ")
                text_without_number = re.sub(r'^\d+(?:\.\d+)*\.\s*', '', formatted).strip()
                if text_without_number or not re.match(r'^\d+(?:\.\d+)*\.\s*$', formatted):
                    # Если есть текст после номера, или это не нумерованный список, добавляем
                    lines.append(formatted)
        elif isinstance(block, Table):
            # При встрече таблицы НЕ сбрасываем счетчики списков,
            # так как Word уже содержит правильную нумерацию
            for row in block.rows:
                # Обрабатываем каждую ячейку отдельно, чтобы сохранить структуру списков
                cell_segments = []
                for cell in row.cells:
                    # Извлекаем текст из ячейки, обрабатывая параграфы внутри ячейки
                    cell_lines = []
                    for para in cell.paragraphs:
                        formatted = _format_paragraph_with_list(para, numbering_state, document)
                        # Пропускаем пустые параграфы (только номер списка без текста)
                        if formatted.strip():
                            # Проверяем, что это не только номер списка
                            text_without_number = re.sub(r'^\d+(?:\.\d+)*\.\s*', '', formatted).strip()
                            if text_without_number or not re.match(r'^\d+(?:\.\d+)*\.\s*$', formatted):
                                cell_lines.append(formatted)
                    cell_text = "\n".join(cell_lines).rstrip()
                    if cell_text:
                        cell_segments.append(cell_text)
                
                # Если в строке несколько ячеек, объединяем их через " | "
                # Если одна ячейка, просто добавляем её содержимое
                if len(cell_segments) > 1:
                    lines.append(" | ".join(cell_segments))
                elif len(cell_segments) == 1:
                    # Для одной ячейки добавляем содержимое как есть (с переводами строк)
                    # Добавляем двойной перевод строки после содержимого ячейки,
                    # чтобы сегментатор мог правильно обработать структурированный текст
                    # как отдельные блоки
                    lines.append(cell_segments[0])
                    # Добавляем пустую строку после содержимого ячейки для разделения блоков
                    lines.append("")
    
    return "\n".join(lines)


def convert_doc_to_txt_mammoth(docx_path: Path, txt_path: Path, encoding: str = "utf-8") -> bool:
    """
    Конвертирует DOCX файл в TXT используя mammoth (если доступен).
    Mammoth лучше сохраняет структуру документа, включая нумерацию списков.
    """
    if not HAS_MAMMOTH:
        return False
    
    try:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text = result.value
        
        # Очищаем текст от лишних пробелов и пустых строк
        lines = text.splitlines()
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Сохраняем только одну пустую строку подряд
                if not prev_empty:
                    cleaned_lines.append("")
                    prev_empty = True
            else:
                cleaned_lines.append(stripped)
                prev_empty = False
        
        # Убираем пустые строки в начале и конце
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(cleaned_lines), encoding=encoding)
        return True
    except Exception as e:
        print(f"Ошибка при конвертации {docx_path} через mammoth: {e}")
        return False


def convert_docs_to_txt(source_dir: Path, target_dir: Path, encoding: str = "utf-8", use_mammoth: bool = True) -> list[Path]:
    """
    Конвертирует все DOCX из source_dir в TXT в target_dir.
    Если use_mammoth=True и mammoth доступен, использует mammoth для лучшего сохранения структуры.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc_path in sorted(source_dir.glob("*.docx")):
        out_path = target_dir / f"{doc_path.stem}.txt"
        
        # Пробуем использовать mammoth, если доступен и запрошен
        if use_mammoth and HAS_MAMMOTH:
            if convert_doc_to_txt_mammoth(doc_path, out_path, encoding):
                written.append(out_path)
                continue
        
        # Fallback: используем python-docx
        document = Document(doc_path)
        text = extract_text(document)
        out_path.write_text(text, encoding=encoding)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация DOCX в TXT без лишних пустых строк.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/specifications/docs"),
        help="Каталог с DOCX-файлами",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/specifications/texts"),
        help="Каталог назначения для TXT",
    )
    parser.add_argument("--encoding", default="utf-8", help="Кодировка выходных файлов")
    args = parser.parse_args()

    written = convert_docs_to_txt(args.src, args.dst, encoding=args.encoding)
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

