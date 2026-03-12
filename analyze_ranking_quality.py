"""
Детальный анализ качества ранжирования обученной cross-encoder модели.
Сравнивает порядок сегментов с bi-encoder и анализирует различия.
"""
from pathlib import Path
from annotation import annotate_document
from annotation.annotator import EmbeddingAnnotator
from annotation.segmenter import TextSegmenter


def analyze_ranking_quality():
    """Анализирует качество ранжирования."""
    base_dir = Path(__file__).parent
    ontology_path = base_dir / "data" / "ontology_filtered.json"
    text_path = base_dir / "data" / "specifications" / "texts" / "2024.05.15_Проект ТЗ_аргон.txt"
    trained_model_path = base_dir / "models" / "cross-encoder-rusbeir" / "final_model"
    
    print("=" * 80)
    print("Детальный анализ ранжирования")
    print("=" * 80)
    print()
    
    # Загружаем текст и сегментируем
    text = text_path.read_text(encoding="utf-8")
    segmenter = TextSegmenter()
    segments = segmenter.segment(text)
    
    # Создаем аннотатор
    annotator_bi = EmbeddingAnnotator(ontology_path=ontology_path, cross_encoder_model=None)
    annotator_cross = EmbeddingAnnotator(ontology_path=ontology_path, cross_encoder_model=str(trained_model_path))
    
    # Получаем аннотации
    annotations_bi = annotator_bi.annotate(text, threshold=0.3, top_k=10, rerank_top_k=0)
    annotations_cross = annotator_cross.annotate(text, threshold=0.3, top_k=10, rerank_top_k=50)
    
    print("Сравнение ранжирования для топ-5 компетенций:\n")
    
    for i in range(min(5, len(annotations_bi), len(annotations_cross))):
        comp_bi = annotations_bi[i]
        comp_cross = annotations_cross[i]
        
        # Находим соответствующую компетенцию в cross-encoder результатах
        comp_cross_match = None
        for ann in annotations_cross:
            if ann['competency_id'] == comp_bi['competency_id']:
                comp_cross_match = ann
                break
        
        if not comp_cross_match:
            continue
        
        print(f"{i+1}. {comp_bi['competency_label']}")
        print(f"   Bi-encoder max score: {comp_bi['max_confidence']:.4f}")
        print(f"   Cross-encoder max score: {comp_cross_match['max_confidence']:.4f}")
        print()
        
        # Сравниваем порядок сегментов
        print("   Порядок сегментов:")
        print("   Bi-encoder:")
        for j, match in enumerate(comp_bi['matches'][:5], 1):
            seg_text = match['segment'].replace('\n', ' ')[:80]
            print(f"     {j}. [{match['score']:.4f}] Сегмент #{match['segment_index']}: {seg_text}...")
        
        print("   Cross-encoder:")
        for j, match in enumerate(comp_cross_match['matches'][:5], 1):
            seg_text = match['segment'].replace('\n', ' ')[:80]
            print(f"     {j}. [{match['score']:.4f}] Сегмент #{match['segment_index']}: {seg_text}...")
        
        # Анализ различий в ранжировании
        bi_segments = [m['segment_index'] for m in comp_bi['matches'][:5]]
        cross_segments = [m['segment_index'] for m in comp_cross_match['matches'][:5]]
        
        # Находим общие сегменты
        common = set(bi_segments) & set(cross_segments)
        only_bi = set(bi_segments) - set(cross_segments)
        only_cross = set(cross_segments) - set(bi_segments)
        
        print(f"   Анализ:")
        print(f"     Общие сегменты в топ-5: {len(common)} ({sorted(common)})")
        if only_bi:
            print(f"     Только в bi-encoder: {sorted(only_bi)}")
        if only_cross:
            print(f"     Только в cross-encoder: {sorted(only_cross)}")
        
        # Проверяем, изменился ли порядок общих сегментов
        if len(common) >= 2:
            bi_order = [s for s in bi_segments if s in common]
            cross_order = [s for s in cross_segments if s in common]
            if bi_order != cross_order:
                print(f"     Порядок общих сегментов изменился:")
                print(f"       Bi-encoder: {bi_order}")
                print(f"       Cross-encoder: {cross_order}")
        
        print()
        print("-" * 80)
        print()
    
    # Анализ распределения скоров
    print("=" * 80)
    print("Анализ распределения скоров")
    print("=" * 80)
    print()
    
    all_bi_scores = []
    all_cross_scores = []
    
    for ann_bi in annotations_bi[:5]:
        for match in ann_bi['matches']:
            all_bi_scores.append(match['score'])
    
    for ann_cross in annotations_cross[:5]:
        for match in ann_cross['matches']:
            all_cross_scores.append(match['score'])
    
    if all_bi_scores and all_cross_scores:
        print(f"Bi-encoder скоры:")
        print(f"  Min: {min(all_bi_scores):.4f}")
        print(f"  Max: {max(all_bi_scores):.4f}")
        print(f"  Mean: {sum(all_bi_scores)/len(all_bi_scores):.4f}")
        print(f"  Std: {(sum((x - sum(all_bi_scores)/len(all_bi_scores))**2 for x in all_bi_scores) / len(all_bi_scores))**0.5:.4f}")
        print()
        print(f"Cross-encoder скоры:")
        print(f"  Min: {min(all_cross_scores):.4f}")
        print(f"  Max: {max(all_cross_scores):.4f}")
        print(f"  Mean: {sum(all_cross_scores)/len(all_cross_scores):.4f}")
        print(f"  Std: {(sum((x - sum(all_cross_scores)/len(all_cross_scores))**2 for x in all_cross_scores) / len(all_cross_scores))**0.5:.4f}")
        print()
        
        # Разброс скоров (разница между max и min)
        bi_range = max(all_bi_scores) - min(all_bi_scores)
        cross_range = max(all_cross_scores) - min(all_cross_scores)
        print(f"Разброс скоров (max - min):")
        print(f"  Bi-encoder: {bi_range:.4f}")
        print(f"  Cross-encoder: {cross_range:.4f}")
        print()
        print(f"Вывод: {'Cross-encoder имеет больший разброс' if cross_range > bi_range else 'Bi-encoder имеет больший разброс'} скоров, что {'лучше' if cross_range > bi_range else 'хуже'} для дискриминации.")


if __name__ == "__main__":
    analyze_ranking_quality()

