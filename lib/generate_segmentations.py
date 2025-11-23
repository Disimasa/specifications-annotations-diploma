"""
Скрипт для генерации JSON-файлов с сегментацией для всех текстовых файлов из папки texts.
"""
from pathlib import Path
import json

from annotation.segmenter import TextSegmenter


def main() -> None:
    base_dir = Path(__file__).parent.parent
    texts_dir = base_dir / "data" / "specifications" / "texts"
    jsons_dir = base_dir / "data" / "segmentations" / "reference"

    # Создаем папку для JSON файлов, если её нет
    jsons_dir.mkdir(parents=True, exist_ok=True)

    # Находим все .txt файлы
    txt_files = list(texts_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"Не найдено .txt файлов в {texts_dir}")
        return

    print(f"Найдено {len(txt_files)} файлов для обработки")
    print(f"Выходная папка: {jsons_dir}\n")

    segmenter = TextSegmenter()

    for txt_file in txt_files:
        print(f"Обработка: {txt_file.name}...")
        
        try:
            # Читаем текст
            text = txt_file.read_text(encoding="utf-8")
            
            # Сегментируем текст
            segments = segmenter.segment(text)

            # Сохраняем JSON файл (просто список строк)
            json_filename = txt_file.stem + ".json"
            json_path = jsons_dir / json_filename
            
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            print(f"  ✓ Сохранено: {json_path.name} ({len(segments)} сегментов)")
            
        except Exception as e:
            print(f"  ✗ Ошибка при обработке {txt_file.name}: {e}")
            continue

    print(f"\nГотово! Обработано {len(txt_files)} файлов.")


if __name__ == "__main__":
    main()

