"""
Тестовый скрипт для модели GrigoryT22/cross-encoder-ru
"""
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from annotation.segmenter import TextSegmenter
from annotation.ontology import Ontology
import json

# Кастомный класс модели из документации
class GrigoryCrossEncoder(nn.Module):
    """
    Cross-encoder модель от GrigoryT22 для русского языка.
    Основана на LaBSE с бинарной классификацией.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.labse_config = AutoConfig.from_pretrained('cointegrated/LaBSE-en-ru')
        self.labse = AutoModel.from_config(self.labse_config)
        self.tokenizer = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')
        self.cls = nn.Sequential(OrderedDict([
            ('dropout_in', torch.nn.Dropout(.0)),
            ('layernorm_in', nn.LayerNorm(768, eps=1e-05)),
            ('fc_1', nn.Linear(768, 768 * 2)),
            ('act_1', nn.GELU()),
            ('layernorm_1', nn.LayerNorm(768 * 2, eps=1e-05)),
            ('fc_2', nn.Linear(768 * 2, 768 * 2)),
            ('act_2', nn.GELU()),
            ('layernorm_2', nn.LayerNorm(768 * 2, eps=1e-05)),
            ('fc_3', nn.Linear(768 * 2, 768)),
            ('act_3', nn.GELU()),
            ('layernorm_3', nn.LayerNorm(768, eps=1e-05)),
            ('fc_4', nn.Linear(768, 256)),
            ('act_4', nn.GELU()),
            ('layernorm_4', nn.LayerNorm(256, eps=1e-05)),
            ('fc_5', nn.Linear(256, 2, bias=True)),
        ]))
        self.to(device)
        self.eval()

    def forward(self, text_pairs):
        """
        text_pairs: список пар [text1, text2]
        Возвращает logits для бинарной классификации [не релевантно, релевантно]
        """
        # Объединяем пары текстов для cross-encoder
        # LaBSE tokenizer принимает пары через специальный формат
        encoded = self.tokenizer(
            [pair[0] for pair in text_pairs],
            [pair[1] for pair in text_pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        model_output = self.labse(**encoded)
        result = self.cls(model_output.pooler_output)
        return result

    def predict(self, text_pairs):
        """
        Предсказывает scores для пар текстов.
        Возвращает logits релевантности (второй класс) или вероятности.
        """
        with torch.no_grad():
            logits = self.forward(text_pairs)
            # Модель возвращает logits для бинарной классификации [не релевантно, релевантно]
            # Возвращаем logit релевантности (второй класс) - это raw score
            relevance_logits = logits[:, 1].cpu().numpy()
            
            # Также можно вернуть вероятности через sigmoid (для бинарной классификации)
            # или через softmax
            probs = torch.softmax(logits, dim=1)
            relevance_probs = probs[:, 1].cpu().numpy()
            
            # Возвращаем и logits, и probabilities для анализа
            return relevance_logits, relevance_probs


def load_model():
    """Загружает модель GrigoryT22/cross-encoder-ru"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    model = GrigoryCrossEncoder(device=device)
    
    # Загружаем веса модели
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="GrigoryT22/cross-encoder-ru",
        filename="model.pt",
        cache_dir=None
    )
    
    print(f"Загрузка весов модели из {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Модель загружена успешно!")
    
    return model


def main():
    base_dir = Path(__file__).parent
    
    # Загружаем данные
    ontology_path = base_dir / "data" / "ontology_filtered.json"
    text_path = base_dir / "data" / "specifications" / "texts" / "2024.05.15_Проект ТЗ_аргон.txt"
    
    print("Загрузка онтологии...")
    with ontology_path.open("r", encoding="utf-8") as f:
        ontology_data = json.load(f)
    ontology = Ontology(ontology_data)
    
    print("Загрузка и сегментация текста...")
    text = text_path.read_text(encoding="utf-8")
    segmenter = TextSegmenter()
    segments = segmenter.segment(text)
    print(f"Получено {len(segments)} сегментов")
    
    # Загружаем модель
    print("\nЗагрузка модели GrigoryT22/cross-encoder-ru...")
    model = load_model()
    
    # Выбираем несколько компетенций для тестирования
    test_competencies = [
        comp for comp in ontology.competencies 
        if any(keyword in comp.label.lower() for keyword in ['литье', 'металлургия', 'модель', 'анализ'])
    ][:3]  # Берем первые 3
    
    print(f"\nТестирование на {len(test_competencies)} компетенциях:")
    for comp in test_competencies:
        print(f"  - {comp.label}")
    
    # Тестируем каждую компетенцию
    for comp in test_competencies:
        print(f"\n{'='*80}")
        print(f"Компетенция: {comp.label}")
        if comp.description:
            print(f"Описание: {comp.description}")
        print(f"{'='*80}")
        
        # Формируем текст компетенции
        comp_text = f"{comp.label}. {comp.description}".strip() if comp.description else comp.label
        
        # Берем первые 10 сегментов для тестирования
        test_segments = segments[:10]
        
        # Создаем пары (компетенция, сегмент)
        pairs = [[comp_text, seg] for seg in test_segments]
        
        # Получаем scores
        print(f"\nВычисление scores для {len(pairs)} пар...")
        logits, probs = model.predict(pairs)
        
        # Сортируем по убыванию score (используем logits, так как они более информативны)
        results = list(zip(test_segments, logits, probs))
        results.sort(key=lambda x: x[1], reverse=True)  # Сортируем по logits
        
        print(f"\nTop {min(5, len(results))} сегментов (logit, probability):")
        for i, (seg, logit, prob) in enumerate(results[:5], 1):
            seg_preview = seg.replace("\n", " ")
            print(f"  {i}. logit={logit:.4f}, prob={prob:.4f}: {seg_preview}")
        
        # Статистика
        print(f"\nСтатистика scores:")
        print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        print(f"  Probabilities: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")


if __name__ == "__main__":
    main()

