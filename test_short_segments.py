"""
Тестирование, почему модель дает высокие scores для коротких сегментов
"""
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json

# Кастомный класс модели
class GrigoryCrossEncoder(nn.Module):
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
        with torch.no_grad():
            logits = self.forward(text_pairs)
            probs = torch.softmax(logits, dim=1)
            return logits[:, 1].cpu().numpy(), probs[:, 1].cpu().numpy()


def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GrigoryCrossEncoder(device=device)
    
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="GrigoryT22/cross-encoder-ru",
        filename="model.pt",
        cache_dir=None
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    base_dir = Path(__file__).parent
    ontology_path = base_dir / "data" / "ontology_filtered.json"
    
    # Загружаем компетенцию
    with ontology_path.open("r", encoding="utf-8") as f:
        ontology_data = json.load(f)
    
    from annotation.ontology import Ontology
    ontology = Ontology(ontology_data)
    
    # Находим компетенцию про литье
    comp = None
    for c in ontology.competencies:
        if "Литье по выплавляемым моделям" in c.label:
            comp = c
            break
    
    comp_text = f"{comp.label}. {comp.description}".strip() if comp.description else comp.label
    print(f"Компетенция: {comp_text}\n")
    
    # Загружаем модель
    print("Загрузка модели...")
    model = load_model()
    
    # Тестируем разные типы сегментов
    test_cases = [
        ("Короткий общий текст", "РАЗДЕЛ 1."),
        ("Очень короткий", "ТЕХНИЧЕСКОЕ ЗАДАНИЕ"),
        ("Один заголовок", "Задачи:"),
        ("Релевантный текст", "Разработка технологии литья по выплавляемым моделям для производства деталей"),
        ("Длинный нерелевантный", "создание математических моделей оценки уровня интенсивности перемешивания металла в сталь-ковше"),
        ("Средний нерелевантный", "адаптация моделей и введение в промышленную эксплуатацию."),
        ("Пустой текст", ""),
        ("Только пробелы", "   "),
    ]
    
    print("="*80)
    print("Тестирование разных типов сегментов:")
    print("="*80)
    
    pairs = [[comp_text, seg] for _, seg in test_cases]
    logits, probs = model.predict(pairs)
    
    results = list(zip(test_cases, logits, probs))
    results.sort(key=lambda x: x[2], reverse=True)  # Сортируем по probability
    
    for (name, seg), logit, prob in results:
        print(f"\n{name}:")
        print(f"  Сегмент: '{seg}'")
        print(f"  Logit: {logit:.4f}, Probability: {prob:.4f}")
    
    print("\n" + "="*80)
    print("Гипотезы:")
    print("="*80)
    print("1. Модель обучена на RAG (вопрос-ответ), где короткие общие фразы могут быть релевантны")
    print("2. Короткие тексты могут давать более 'универсальные' эмбеддинги")
    print("3. Модель может быть переобучена на определенные паттерны в обучающих данных")
    print("4. LaBSE (базовая модель) может по-разному обрабатывать короткие и длинные тексты")


if __name__ == "__main__":
    main()

