"""
Скрипт для тестирования обученной cross-encoder модели на файле аргона.
Показывает скоры для каждого сегмента.
"""
import json
from pathlib import Path

from annotation import annotate_document
from annotation.annotator import EmbeddingAnnotator
from annotation.segmenter import TextSegmenter


def test_trained_model_on_argon():
    """Тестирует обученную модель на файле аргона и выводит скоры."""
    base_dir = Path(__file__).parent
    ontology_path = base_dir / "data" / "ontology_filtered.json"
    text_path = base_dir / "data" / "specifications" / "texts" / "2024.05.15_Проект ТЗ_аргон.txt"
    
    # Путь к обученной модели
    trained_model_path = base_dir / "models" / "cross-encoder-rusbeir" / "final_model"
    
    print("=" * 80)
    print("Тестирование обученной cross-encoder модели на файле аргона")
    print("=" * 80)
    print(f"Модель: {trained_model_path}")
    print(f"Файл: {text_path.name}")
    print()
    
    # Загружаем текст и сегментируем
    text = text_path.read_text(encoding="utf-8")
    segmenter = TextSegmenter()
    segments = segmenter.segment(text)
    
    print(f"Всего сегментов: {len(segments)}")
    print()
    
    # Тестируем с обученной моделью
    print("=" * 80)
    print("Аннотация с обученной cross-encoder моделью")
    print("=" * 80)
    print()
    
    annotations = annotate_document(
        text_path,
        ontology_path,
        threshold=0.3,  # Понижаем порог для демонстрации
        top_k=5,
        rerank_top_k=50,  # Используем re-ranking
        cross_encoder_model=str(trained_model_path)  # Используем обученную модель
    )
    
    print(f"Найдено компетенций: {len(annotations)}")
    print()
    
    # Выводим результаты с детальными скорами
    for i, ann in enumerate(annotations[:10], 1):  # Показываем топ-10
        print(f"{i}. {ann['competency_label']}")
        print(f"   Максимальный скор: {ann['max_confidence']:.4f}")
        print(f"   Найдено сегментов: {len(ann['matches'])}")
        print()
        
        for j, match in enumerate(ann['matches'], 1):
            segment_idx = match['segment_index']
            score = match['score']
            segment_text = match['segment'].replace('\n', ' ').strip()
            
            # Обрезаем длинный текст
            if len(segment_text) > 150:
                segment_text = segment_text[:150] + "..."
            
            print(f"   {j}. Скор: {score:.4f} | Сегмент #{segment_idx}")
            print(f"      {segment_text}")
            print()
        
        print("-" * 80)
        print()
    
    # Также покажем сравнение: без re-ranking vs с re-ranking
    print("=" * 80)
    print("Сравнение: БЕЗ re-ranking vs С re-ranking (обученная модель)")
    print("=" * 80)
    print()
    
    # Без re-ranking
    annotations_no_rerank = annotate_document(
        text_path,
        ontology_path,
        threshold=0.3,
        top_k=5,
        rerank_top_k=0  # Без re-ranking
    )
    
    # С re-ranking (обученная модель)
    annotations_with_rerank = annotate_document(
        text_path,
        ontology_path,
        threshold=0.3,
        top_k=5,
        rerank_top_k=50,
        cross_encoder_model=str(trained_model_path)
    )
    
    print("Топ-5 компетенций БЕЗ re-ranking:")
    for i, ann in enumerate(annotations_no_rerank[:5], 1):
        print(f"  {i}. {ann['competency_label']} (max: {ann['max_confidence']:.4f})")
    
    print()
    print("Топ-5 компетенций С re-ranking (обученная модель):")
    for i, ann in enumerate(annotations_with_rerank[:5], 1):
        print(f"  {i}. {ann['competency_label']} (max: {ann['max_confidence']:.4f})")
    
    print()
    print("=" * 80)
    print("Детальный вывод скоров для первой компетенции")
    print("=" * 80)
    print()
    
    if annotations_with_rerank:
        first_comp = annotations_with_rerank[0]
        print(f"Компетенция: {first_comp['competency_label']}")
        print(f"Всего сегментов: {len(first_comp['matches'])}")
        print()
        
        print("Скоры сегментов (отсортированы по убыванию):")
        for match in first_comp['matches']:
            print(f"  Сегмент #{match['segment_index']}: {match['score']:.4f}")
            segment_text = match['segment'].replace('\n', ' ').strip()
            if len(segment_text) > 100:
                segment_text = segment_text[:100] + "..."
            print(f"    {segment_text}")
            print()


if __name__ == "__main__":
    test_trained_model_on_argon()

