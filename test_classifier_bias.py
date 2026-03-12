"""
Проверка смещения классификатора
"""
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np

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
        pooler_output = model_output.pooler_output
        result = self.cls(pooler_output)
        return result, pooler_output


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
    model = load_model()
    
    # Тестируем с разными парами
    comp_text = "Литье по выплавляемым моделям. Высокоточное литье по восковым моделям"
    
    test_pairs = [
        [comp_text, "РАЗДЕЛ 1."],
        [comp_text, "ТЕХНИЧЕСКОЕ ЗАДАНИЕ"],
        [comp_text, ""],
        [comp_text, "Разработка технологии литья по выплавляемым моделям"],
    ]
    
    print("Анализ работы классификатора:")
    print("="*80)
    
    for pair in test_pairs:
        with torch.no_grad():
            logits, pooler_emb = model.forward([pair])
            logits = logits[0].cpu().numpy()
            pooler_emb = pooler_emb[0].cpu().numpy()
            
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            
            print(f"\nПара: '{pair[0][:40]}...' <-> '{pair[1]}'")
            print(f"  Pooler embedding norm: {np.linalg.norm(pooler_emb):.4f}")
            print(f"  Logits: [не релевантно={logits[0]:.4f}, релевантно={logits[1]:.4f}]")
            print(f"  Probabilities: [не релевантно={probs[0]:.4f}, релевантно={probs[1]:.4f}]")
            print(f"  Разница logits: {logits[1] - logits[0]:.4f}")
    
    # Проверяем bias в последнем слое
    print("\n" + "="*80)
    print("Анализ bias в последнем слое классификатора:")
    print("="*80)
    
    last_layer = model.cls.fc_5
    bias = last_layer.bias.cpu().detach().numpy()
    print(f"Bias: [не релевантно={bias[0]:.4f}, релевантно={bias[1]:.4f}]")
    print(f"Разница bias: {bias[1] - bias[0]:.4f}")
    
    if bias[1] > bias[0]:
        print("\n✓ Классификатор имеет положительный bias для класса 'релевантно'")
        print("  Это объясняет, почему модель склонна выдавать высокие scores")


if __name__ == "__main__":
    main()

