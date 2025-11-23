from pathlib import Path

from annotation import annotate_document


def main() -> None:
    base_dir = Path(__file__).parent
    ontology_path = base_dir / "data" / "ontology_filtered.json"
    text_path = base_dir / "data" / "specifications" / "texts" / "ТЗ_1518025 Определение характеристик мелющих шаров_2023_ТЗ №4.txt"

    annotations = annotate_document(text_path, ontology_path, threshold=0.45, top_k=10)
    for ann in annotations:
        print(f"{ann['competency_label']} (max {ann['max_confidence']:.2f}):")
        for match in ann["matches"]:
            snippet = match["segment"].replace("\n", " ")
            print(f"  - {match['score']:.2f}: {snippet}")
        print()


if __name__ == "__main__":
    main()

