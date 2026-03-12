"""
Скрипт для оценки обученного cross-encoder на тестовых датасетах
"""
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
    CERerankingEvaluator,
)


def load_rusbeir_dataset(hf_repo: str, hf_repo_qrels: str, split: str = "test"):
    """Загружает датасет из RusBEIR формата."""
    print(f"Загрузка датасета {hf_repo} (split={split})...")
    
    # Сначала пробуем загрузить с указанным split
    try:
        corpus_ds = load_dataset(hf_repo, "corpus", split=split)
        queries_ds = load_dataset(hf_repo, "queries", split=split)
        qrels_ds = load_dataset(hf_repo_qrels, split=split)
    except Exception as e:
        # Если split недоступен, пробуем train
        if split == "test":
            print(f"Split 'test' недоступен, пробуем 'train'...")
            try:
                corpus_ds = load_dataset(hf_repo, "corpus", split="train")
                queries_ds = load_dataset(hf_repo, "queries", split="train")
                qrels_ds = load_dataset(hf_repo_qrels, split="train")
            except Exception as e2:
                raise Exception(f"Не удалось загрузить датасет (test и train): {e}, {e2}")
        else:
            raise Exception(f"Не удалось загрузить датасет: {e}")
    
    corpus = {}
    queries = {}
    qrels = {}
    
    for item in corpus_ds:
        doc_id = item.get("_id", item.get("id"))
        text = item.get("text", item.get("title", "") + " " + item.get("text", ""))
        corpus[doc_id] = {"text": text}
    
    for item in queries_ds:
        query_id = item.get("_id", item.get("id"))
        query_text = item.get("text", item.get("query", ""))
        queries[query_id] = query_text
    
    for item in qrels_ds:
        query_id = item.get("query-id", item.get("query_id"))
        doc_id = item.get("corpus-id", item.get("doc_id", item.get("corpus_id")))
        score = item.get("score", 1)
        
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = score
    
    return corpus, queries, qrels


def evaluate_reranking(
    model: CrossEncoder, corpus: Dict, queries: Dict, qrels: Dict, top_k: int = 100, max_queries: int = 50, max_docs_per_query: int = 1000
):
    """
    Оценивает модель в режиме re-ranking.
    Для каждого запроса ранжирует документы и вычисляет метрики.
    
    Args:
        max_queries: Максимальное количество запросов для оценки
        max_docs_per_query: Максимальное количество документов для ранжирования на запрос
    """
    print(f"\nОценка re-ranking (top_k={top_k}, max_queries={max_queries}, max_docs_per_query={max_docs_per_query})...")
    
    all_results = []
    query_items = list(queries.items())[:max_queries]
    
    for idx, (query_id, query_text) in enumerate(query_items, 1):
        if query_id not in qrels:
            continue
        
        query_id_str = str(query_id)[:50] if query_id else "unknown"
        print(f"Обработка запроса {idx}/{len(query_items)}: {query_id_str}...")
        
        # Ограничиваем количество документов для ранжирования
        # Берем релевантные документы + случайные нерелевантные
        relevant_docs = list(qrels[query_id].keys())
        non_relevant_docs = [
            doc_id for doc_id in corpus.keys() if doc_id not in relevant_docs
        ]
        
        # Берем все релевантные + случайные нерелевантные до max_docs_per_query
        import random
        docs_to_rank = relevant_docs.copy()
        remaining_slots = max_docs_per_query - len(docs_to_rank)
        if remaining_slots > 0 and non_relevant_docs:
            docs_to_rank.extend(random.sample(non_relevant_docs, min(remaining_slots, len(non_relevant_docs))))
        
        # Создаем пары (query, document) только для выбранных документов
        sentence_pairs = []
        doc_ids = []
        
        for doc_id in docs_to_rank:
            sentence_pairs.append([query_text, corpus[doc_id]["text"]])
            doc_ids.append(doc_id)
        
        if not sentence_pairs:
            continue
        
        # Получаем scores от модели
        print(f"  Ранжирование {len(sentence_pairs)} документов...")
        scores = model.predict(sentence_pairs, show_progress_bar=True, batch_size=32)
        
        # Сортируем по убыванию score
        ranked_docs = sorted(
            zip(doc_ids, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # Вычисляем метрики
        relevant_docs_set = set(relevant_docs)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]
        
        # Precision@k, Recall@k, NDCG@k
        relevant_retrieved = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs_set)
        
        precision = relevant_retrieved / len(ranked_doc_ids) if ranked_doc_ids else 0
        recall = relevant_retrieved / len(relevant_docs_set) if relevant_docs_set else 0
        
        # NDCG@k (упрощенная версия)
        dcg = sum(
            (1.0 / (i + 1)) if doc_id in relevant_docs_set else 0
            for i, doc_id in enumerate(ranked_doc_ids)
        )
        idcg = sum(1.0 / (i + 1) for i in range(min(len(relevant_docs_set), len(ranked_doc_ids))))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        all_results.append({
            "query_id": query_id,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
        })
        
        print(f"  Precision@{top_k}: {precision:.4f}, Recall@{top_k}: {recall:.4f}, NDCG@{top_k}: {ndcg:.4f}")
    
    if not all_results:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "ndcg@k": 0.0,
        }
    
    # Средние метрики
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall"] for r in all_results) / len(all_results)
    avg_ndcg = sum(r["ndcg"] for r in all_results) / len(all_results)
    
    return {
        "precision@k": avg_precision,
        "recall@k": avg_recall,
        "ndcg@k": avg_ndcg,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка cross-encoder модели")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/cross-encoder-rusbeir",
        help="Путь к обученной модели",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sberquad-retrieval",
        help="Название датасета для оценки",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split для оценки (test/dev)",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Ошибка: модель не найдена в {model_path}")
        return
    
    print(f"Загрузка модели из {model_path}...")
    model = CrossEncoder(str(model_path))
    
    # Загружаем тестовый датасет
    dataset_configs = {
        "sberquad-retrieval": {
            "hf_repo": "kngrg/sberquad-retrieval",
            "hf_repo_qrels": "kngrg/sberquad-retrieval-qrels",
        },
        # Добавьте другие датасеты по необходимости
    }
    
    if args.dataset not in dataset_configs:
        print(f"Ошибка: неизвестный датасет {args.dataset}")
        print(f"Доступные: {list(dataset_configs.keys())}")
        return
    
    config = dataset_configs[args.dataset]
    corpus, queries, qrels = load_rusbeir_dataset(
        config["hf_repo"], config["hf_repo_qrels"], split=args.split
    )
    
    print(f"\nЗагружено: {len(corpus)} документов, {len(queries)} запросов")
    
    # Оценка бинарной классификации
    print("\n" + "=" * 80)
    print("Оценка бинарной классификации")
    print("=" * 80)
    
    # Создаем пары для оценки (выборка для скорости)
    sentence_pairs = []
    labels = []
    
    import random
    sample_queries = random.sample(list(queries.items()), min(50, len(queries)))
    print(f"Выбрано {len(sample_queries)} запросов для оценки бинарной классификации")
    
    for query_id, query_text in sample_queries:
        if query_id not in qrels:
            continue
        
        # Берем релевантные и случайные нерелевантные документы
        relevant_docs = list(qrels[query_id].keys())
        non_relevant_docs = [
            doc_id for doc_id in corpus.keys() if doc_id not in relevant_docs
        ]
        
        # Добавляем релевантные
        for doc_id in relevant_docs[:5]:  # Ограничиваем
            sentence_pairs.append([query_text, corpus[doc_id]["text"]])
            labels.append(1.0)
        
        # Добавляем нерелевантные
        for doc_id in random.sample(non_relevant_docs, min(5, len(non_relevant_docs))):
            sentence_pairs.append([query_text, corpus[doc_id]["text"]])
            labels.append(0.0)
    
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=sentence_pairs,
        labels=labels,
        name=args.dataset,
        show_progress_bar=True,
    )
    
    results = evaluator(model)
    print("\nРезультаты:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Оценка re-ranking
    print("\n" + "=" * 80)
    print("Оценка re-ranking")
    print("=" * 80)
    
    rerank_results = evaluate_reranking(model, corpus, queries, qrels, top_k=10, max_queries=20, max_docs_per_query=500)
    print("\nРезультаты re-ranking:")
    for metric, value in rerank_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Сохраняем результаты
    results_path = model_path / f"evaluation_{args.dataset}_{args.split}.json"
    all_results = {
        "dataset": args.dataset,
        "split": args.split,
        "binary_classification": results,
        "reranking": rerank_results,
    }
    
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {results_path}")


if __name__ == "__main__":
    main()

