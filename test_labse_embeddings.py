"""
Проверка эмбеддингов LaBSE для коротких текстов
"""
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained('cointegrated/LaBSE-en-ru')
tokenizer = AutoTokenizer.from_pretrained('cointegrated/LaBSE-en-ru')
model.to(device)
model.eval()

test_texts = [
    "РАЗДЕЛ 1.",
    "ТЕХНИЧЕСКОЕ ЗАДАНИЕ",
    "Задачи:",
    "",
    "   ",
    "Разработка технологии литья по выплавляемым моделям для производства деталей",
    "Литье по выплавляемым моделям. Высокоточное литье по восковым моделям и керамическим оболочкам",
]

print("Анализ эмбеддингов LaBSE для разных текстов:")
print("="*80)

embeddings = []
for text in test_texts:
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**encoded)
        # Используем pooler_output (как в модели)
        emb = output.pooler_output.cpu().numpy()[0]
        embeddings.append(emb)
        
        # Норма вычисляем норму эмбеддинга
        norm = np.linalg.norm(emb)
        print(f"\n'{text[:50]}':")
        print(f"  Норма эмбеддинга: {norm:.4f}")
        print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}, Mean: {emb.mean():.4f}")

# Проверяем косинусное сходство между эмбеддингами
print("\n" + "="*80)
print("Косинусное сходство между эмбеддингами:")
print("="*80)

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)

for i, text1 in enumerate(test_texts):
    for j, text2 in enumerate(test_texts):
        if i < j:
            sim = similarity_matrix[i][j]
            print(f"'{text1[:30]}' <-> '{text2[:30]}': {sim:.4f}")

