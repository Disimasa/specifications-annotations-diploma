"""
Детальный анализ эмбеддингов и классификатора
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

    def forward(self, text_pairs, return_intermediate=False):
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
        
        if return_intermediate:
            # Проходим через классификатор пошагово
            x = pooler_output
            intermediate = {'pooler': x.cpu().numpy()}
            
            x = self.cls.layernorm_in(x)
            intermediate['after_layernorm_in'] = x.cpu().numpy()
            
            x = self.cls.fc_1(x)
            intermediate['after_fc_1'] = x.cpu().numpy()
            
            x = self.cls.act_1(x)
            intermediate['after_act_1'] = x.cpu().numpy()
            
            x = self.cls.layernorm_1(x)
            intermediate['after_layernorm_1'] = x.cpu().numpy()
            
            # ... и так далее до конца
            result = self.cls(pooler_output)
            return result, intermediate
        
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
    
    comp_text = "Литье по выплавляемым моделям. Высокоточное литье по восковым моделям"
    
    test_pairs = [
        ("Короткий", comp_text, "РАЗДЕЛ 1."),
        ("Релевантный", comp_text, "Разработка технологии литья по выплавляемым моделям"),
        ("Пустой", comp_text, ""),
    ]
    
    print("Анализ работы модели:")
    print("="*80)
    
    for name, text1, text2 in test_pairs:
        with torch.no_grad():
            logits, pooler_emb = model.forward([[text1, text2]])
            logits = logits[0].cpu().numpy()
            pooler_emb = pooler_emb[0].cpu().numpy()
            
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            
            print(f"\n{name}: '{text2}'")
            print(f"  Pooler embedding: norm={np.linalg.norm(pooler_emb):.4f}, mean={pooler_emb.mean():.4f}")
            print(f"  Logits: [{logits[0]:.4f}, {logits[1]:.4f}], diff={logits[1]-logits[0]:.4f}")
            print(f"  Probability релевантно: {probs[1]:.4f}")
    
    # Проверяем веса последнего слоя
    print("\n" + "="*80)
    print("Анализ весов последнего слоя:")
    print("="*80)
    
    last_layer = model.cls.fc_5
    weights = last_layer.weight.cpu().detach().numpy()
    bias = last_layer.bias.cpu().detach().numpy()
    
    print(f"Веса для класса 'не релевантно': mean={weights[0].mean():.4f}, std={weights[0].std():.4f}")
    print(f"Веса для класса 'релевантно': mean={weights[1].mean():.4f}, std={weights[1].std():.4f}")
    print(f"Bias: [{bias[0]:.4f}, {bias[1]:.4f}]")
    
    # Проверяем, что происходит с входом в последний слой
    print("\n" + "="*80)
    print("Анализ входа в последний слой:")
    print("="*80)
    
    for name, text1, text2 in test_pairs:
        with torch.no_grad():
            encoded = model.tokenizer(
                [text1], [text2],
                padding=True, truncation=True, max_length=512, return_tensors='pt'
            ).to(model.device)
            
            model_output = model.labse(**encoded)
            pooler = model_output.pooler_output
            
            # Проходим через все слои до последнего
            x = model.cls.layernorm_in(pooler)
            x = model.cls.fc_1(x)
            x = model.cls.act_1(x)
            x = model.cls.layernorm_1(x)
            x = model.cls.fc_2(x)
            x = model.cls.act_2(x)
            x = model.cls.layernorm_2(x)
            x = model.cls.fc_3(x)
            x = model.cls.act_3(x)
            x = model.cls.layernorm_3(x)
            x = model.cls.fc_4(x)
            x = model.cls.act_4(x)
            x = model.cls.layernorm_4(x)
            
            # Это вход в последний слой
            input_to_last = x.cpu().numpy()[0]
            
            # Вычисляем, что даст последний слой
            output_0 = np.dot(weights[0], input_to_last) + bias[0]
            output_1 = np.dot(weights[1], input_to_last) + bias[1]
            
            print(f"\n{name}: '{text2}'")
            print(f"  Вход в последний слой: norm={np.linalg.norm(input_to_last):.4f}, mean={input_to_last.mean():.4f}")
            print(f"  Выход слоя: [{output_0:.4f}, {output_1:.4f}], diff={output_1-output_0:.4f}")


if __name__ == "__main__":
    main()

