from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import List

try:
    import aspose.words as aw
    HAS_ASPOSE = True
except ImportError:
    HAS_ASPOSE = False

# Флаг для отслеживания, была ли лицензия загружена
_license_loaded = False


def _load_license() -> None:
    """
    Загружает лицензию aspose.words из папки keys.
    """
    global _license_loaded
    if _license_loaded or not HAS_ASPOSE:
        return
    
    try:
        # Путь к лицензии относительно корня проекта
        license_path = Path(__file__).parent.parent / "keys" / "Aspose.WordsforPythonvia.NET.lic"
        
        if license_path.exists():
            license = aw.License()
            license.set_license(str(license_path))
            _license_loaded = True
        # Не выводим сообщения, если лицензия не найдена - это нормально для разработки
    except Exception:
        # Игнорируем ошибки загрузки лицензии
        pass


def _remove_watermarks(document: aw.Document) -> None:
    """
    Удаляет водяные знаки из документа aspose.words.
    Водяные знаки обычно находятся в заголовках и футерах.
    """
    try:
        # Удаляем водяные знаки из всех секций
        for section in document.sections:
            # Удаляем водяные знаки из заголовков и футеров
            for header_footer in section.headers_footers:
                # Получаем все узлы из заголовка/футера
                nodes = header_footer.get_child_nodes(aw.NodeType.ANY, True)
                
                # Собираем узлы для удаления
                nodes_to_remove = []
                for node in nodes:
                    # Удаляем Shape объекты (могут быть водяными знаками)
                    if node.node_type == aw.NodeType.SHAPE:
                        nodes_to_remove.append(node)
                    # Удаляем GroupShape (группы фигур, могут содержать водяные знаки)
                    elif node.node_type == aw.NodeType.GROUP_SHAPE:
                        nodes_to_remove.append(node)
                    # Удаляем параграфы, которые могут содержать только водяные знаки
                    elif node.node_type == aw.NodeType.PARAGRAPH:
                        para = node
                        # Если параграф содержит только Shape или GroupShape, удаляем его
                        has_only_shapes = True
                        for child in para.get_child_nodes(aw.NodeType.ANY, False):
                            if child.node_type not in (aw.NodeType.SHAPE, aw.NodeType.GROUP_SHAPE, aw.NodeType.RUN):
                                has_only_shapes = False
                                break
                        if has_only_shapes:
                            nodes_to_remove.append(node)
                
                # Удаляем собранные узлы
                for node in nodes_to_remove:
                    try:
                        node.remove()
                    except Exception:
                        # Если не удалось удалить, пропускаем
                        pass
    except Exception:
        # Если не удалось удалить водяные знаки, продолжаем работу
        pass


def _remove_aspose_watermarks(text: str) -> str:
    """
    Удаляет текстовые водяные знаки aspose.words из текста.
    """
    # Паттерны для удаления водяных знаков aspose.words
    watermark_patterns = [
        r'Created with an evaluation copy of Aspose\.Words\..*?license/',
        r'Evaluation Only\. Created with Aspose\.Words\..*?Aspose Pty Ltd\.',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        # Проверяем, не является ли строка водяным знаком
        is_watermark = False
        for pattern in watermark_patterns:
            if re.search(pattern, line, re.IGNORECASE | re.DOTALL):
                is_watermark = True
                break
        
        # Удаляем строки, содержащие только цифры в начале файла (первые 3 строки)
        # Это могут быть артефакты от нумерации страниц или водяных знаков
        if i < 3 and re.match(r'^[\s]*\d+[\s]*$', line):
            is_watermark = True
        
        if not is_watermark:
            cleaned_lines.append(line)
    
    # Убираем пустые строки в начале и конце
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    # Убираем BOM и другие невидимые символы в начале
    if cleaned_lines and cleaned_lines[0]:
        cleaned_lines[0] = cleaned_lines[0].lstrip('\ufeff\u200b\u200c\u200d\ufeff')
        # Если первая строка после очистки содержит только цифры и пробелы, удаляем её
        if re.match(r'^[\s]*\d+[\s]*$', cleaned_lines[0]):
            cleaned_lines.pop(0)
    
    return '\n'.join(cleaned_lines)


def _normalize_whitespace(text: str) -> str:
    """
    Нормализует пробельные символы, заменяя NBSP и другие неразрывные пробелы на обычные.
    Это важно для корректной работы сегментации и аннотации.
    """
    # Заменяем различные типы неразрывных пробелов на обычные пробелы
    # \u00A0 - NBSP (non-breaking space)
    # \u2007 - figure space
    # \u202F - narrow no-break space
    # \u2060 - word joiner
    # \uFEFF - zero width no-break space (BOM)
    non_breaking_spaces = '\u00A0\u2007\u202F\u2060\uFEFF'
    
    for nbsp in non_breaking_spaces:
        text = text.replace(nbsp, ' ')
    
    return text


def _normalize_bullet_markers(text: str) -> str:
    """
    Нормализует пробелы вокруг маркеров ненумерованных списков.
    Маркеры оставляем как есть, чтобы сегментатор мог их правильно распознать.
    Только нормализуем пробелы для единообразия.
    """
    # Не заменяем маркеры, только нормализуем пробелы вокруг них
    # Это позволяет сегментатору правильно распознать разные типы списков
    lines = text.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Нормализуем множественные пробелы после маркеров
        # Оставляем один пробел после маркера
        normalized_line = re.sub(r'([•◦▪▫○●◉◯◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡·\u00B7o\-])\s{2,}', r'\1 ', line)
        normalized_lines.append(normalized_line)
    
    return '\n'.join(normalized_lines)


def extract_text(docx_path: Path) -> str:
    """
    Извлекает текст из DOCX файла используя aspose.words.
    Удаляет водяные знаки и нормализует маркеры списков.
    """
    if not HAS_ASPOSE:
        raise ImportError("aspose.words не установлен. Установите: pip install aspose-words")
    
    # Загружаем лицензию (если еще не загружена)
    _load_license()
    
    # Загружаем документ
    doc = aw.Document(str(docx_path))
    
    # Удаляем водяные знаки
    _remove_watermarks(doc)
    
    # Сохраняем во временный TXT файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Сохраняем в TXT
        doc.save(tmp_path, aw.SaveFormat.TEXT)
        
        # Читаем результат
        with open(tmp_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Удаляем водяные знаки aspose.words
        text = _remove_aspose_watermarks(text)
        
        # Нормализуем пробельные символы (заменяем NBSP на обычные пробелы)
        text = _normalize_whitespace(text)
        
        # Нормализуем маркеры ненумерованных списков
        text = _normalize_bullet_markers(text)
        
        return text
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def convert_docs_to_txt(source_dir: Path, target_dir: Path, encoding: str = "utf-8") -> List[Path]:
    """
    Конвертирует все DOCX из source_dir в TXT в target_dir используя aspose.words.
    
    Args:
        source_dir: Каталог с DOCX файлами
        target_dir: Каталог назначения для TXT файлов
        encoding: Кодировка выходных файлов
    """
    
    target_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    
    for doc_path in sorted(source_dir.glob("*.docx")):
        try:
            text = extract_text(doc_path)
            out_path = target_dir / f"{doc_path.stem}.txt"
            out_path.write_text(text, encoding=encoding)
            written.append(out_path)
        except Exception as e:
            print(f"Ошибка при конвертации {doc_path}: {e}")
    
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация DOCX в TXT используя aspose.words.")
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
