"""Тестовый скрипт для показа cross-encoder скоров на файле аргона."""
from pathlib import Path
from annotation import annotate_document

base_dir = Path(__file__).parent
ontology_path = base_dir / "data" / "ontology_filtered.json"
text_path = base_dir / "data" / "specifications" / "texts" / "2024.05.15_Проект ТЗ_аргон.txt"

print("=" * 80)
print("Cross-encoder скоры для файла аргона")
print("=" * 80)
print()

# Используем re-ranking с cross-encoder
annotations = annotate_document(
    text_path,
    ontology_path,
    threshold=0.001,  # Низкий порог, чтобы увидеть все скоры
    top_k=10,
    rerank_top_k=30,  # Используем re-ranking для топ-30 компетенций
)

print(f"Найдено компетенций: {len(annotations)}")
print()

# Показываем все компетенции с их cross-encoder скорами
for i, ann in enumerate(annotations, 1):
    print(f"{i}. {ann['competency_label']} (max_confidence: {ann['max_confidence']:.4f})")
    print(f"   Найдено сегментов: {len(ann['matches'])}")
    print()
    
    # Показываем все сегменты с их cross-encoder скорами
    for j, match in enumerate(ann['matches'], 1):
        segment_text = match['segment'].replace('\n', ' ').strip()
        if len(segment_text) > 100:
            segment_text = segment_text[:100] + "..."
        
        print(f"   {j}. Скор: {match['score']:.6f} | Сегмент #{match['segment_index']}")
        print(f"      {segment_text}")
        print()
    
    print("-" * 80)
    print()

