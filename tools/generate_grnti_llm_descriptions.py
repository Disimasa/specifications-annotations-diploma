from __future__ import annotations

"""
Генерация LLM-описаний для листовых узлов ontology_grnti_clean.json
с помощью OpenRouter (модель qwen/qwen3-vl-235b-a22b-thinking).

- Использует ключ OPENROUTER_API_KEY из .env
- Добавляет поле "llm_description" для листовых нод, где его ещё нет
- Пишет результат в data/ontology_grnti_with_llm.json
- После каждого успешного запроса сохраняет файл (чтобы не терять прогресс)
- Есть ретраи и таймауты на HTTP-запросы
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_clean.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"

ROOT_ID = "http://example.org/grnti_root"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("grnti_llm")


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не найден в .env")
    return api_key


def is_leaf(node_id: str, links: List[dict]) -> bool:
    return not any(link.get("source") == node_id for link in links)


def build_nodes_by_id(nodes: List[dict]) -> Dict[str, dict]:
    return {n["id"]: n for n in nodes if "id" in n}


def generate_batch_descriptions_with_openrouter(
    api_key: str,
    full_labels: List[str],
    timeout: int = 120,
    max_retries: int = 5,
    base_sleep: float = 2.0,
) -> List[Optional[str]]:
    """
    Вызывает OpenRouter (Qwen) ОДНИМ запросом для партии рубрик.
    На вход: список full_label (без кодов).
    На выход: список описаний той же длины (или None, если парсинг для элемента не удался).
    """
    if not full_labels:
        return []

    system_prompt = (
        "Ты — эксперт по классификатору научно-технических рубрик (ГРНТИ). "
        "Говоришь по-русски, используешь максимально специализированную терминологию."
    )

    # Нумеруем рубрики, чтобы модель вернула такой же список
    items_block = "\n".join(f"{i+1}) {fl}" for i, fl in enumerate(full_labels))

    user_prompt = f"""
Тебе даётся список рубрик ГРНТИ, каждая на отдельной строке в формате "N) Текст рубрики".

{items_block}

Для КАЖДОЙ рубрики сгенерируй ОДНО предложение на русском языке длиной 20–25 слов
с максимально специализированной терминологией, описывающее эту рубрику.

Требования:
- не добавляй новые области применения по сравнению с текстом рубрики;
- не повторяй дословно текст рубрики, перефразируй и конкретизируй;
- не повторяй служебные слова "Раздел", "Область" и т.п., используй содержательные термины;
- ответ верни в виде N строк формата "N) сгенерированное_предложение";
- не добавляй никаких пояснений, комментариев или лишнего текста.
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://grnti-local-generator",
        "X-Title": "grnti-ontology-description-generator",
    }

    payload: Dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 429:
                # rate limit — подождать и повторить
                last_err = RuntimeError(f"429 Too Many Requests: {resp.text}")
            else:
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"Пустой ответ OpenRouter: {data}")
                content = choices[0]["message"]["content"]
                if isinstance(content, list):
                    text_parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
                    text = "\n".join(text_parts)
                else:
                    text = str(content)

                # Разбираем по строкам "N) ..."
                results: List[Optional[str]] = [None] * len(full_labels)
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Ищем префикс "N)"
                    if ")" in line:
                        prefix, rest = line.split(")", 1)
                        prefix = prefix.strip()
                        rest = rest.strip()
                        if prefix.isdigit():
                            idx = int(prefix) - 1
                            if 0 <= idx < len(results):
                                results[idx] = rest
                # Если какие-то описания не распарсились, оставляем None
                return results
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, RuntimeError) as exc:
            last_err = exc
            if attempt == max_retries:
                logger.error("LLM-запрос окончательно провалился (попытка %d/%d): %s", attempt, max_retries, exc)
                raise
            sleep_for = base_sleep * attempt
            logger.warning(
                "Ошибка LLM-запроса (попытка %d/%d): %s. Повтор через %.1f с",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    raise RuntimeError(f"Не удалось получить описания из OpenRouter: {last_err}")


def main() -> None:
    api_key = load_api_key()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл {INPUT_PATH}")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    nodes: List[dict] = data.get("nodes", [])
    links: List[dict] = data.get("links", [])

    nodes_by_id = build_nodes_by_id(nodes)

    # Загружаем предыдущие результаты, если файл уже есть —
    # чтобы можно было дозаполнять описания частями.
    if OUTPUT_PATH.exists():
        logger.info("Найден существующий %s, загружаем для продолжения.", OUTPUT_PATH)
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        existing_nodes_by_id = {n["id"]: n for n in existing.get("nodes", []) if "id" in n}
        for node in nodes:
            nid = node.get("id")
            if nid in existing_nodes_by_id:
                llm_desc = existing_nodes_by_id[nid].get("llm_description")
                if llm_desc:
                    node["llm_description"] = llm_desc

    # Находим все листовые компетенции (id с компетенциями + нет исходящих ссылок)
    leaf_nodes = [
        n
        for n in nodes
        if n.get("id", "").startswith("http://example.org/competencies#") and is_leaf(n["id"], links)
    ]

    logger.info("Всего листовых нод: %d", len(leaf_nodes))

    to_generate = [n for n in leaf_nodes if not n.get("llm_description")]
    logger.info("Нод без llm_description: %d", len(to_generate))

    if not to_generate:
        logger.info("Все листовые ноды уже имеют llm_description, делать нечего.")
        return

    batch_size = 50
    for i in tqdm(range(0, len(to_generate), batch_size), desc="Генерация описаний для листовых нод (батчи)"):
        batch = to_generate[i : i + batch_size]
        full_labels = [(n.get("full_label") or n.get("label") or "").strip() for n in batch]

        try:
            descriptions = generate_batch_descriptions_with_openrouter(
                api_key=api_key,
                full_labels=full_labels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка генерации для батча %d-%d: %s", i, i + len(batch) - 1, exc)
            # Пропускаем этот батч и идём дальше, прогресс всё равно сохранён для предыдущих
            continue

        for node, desc in zip(batch, descriptions):
            if not desc:
                continue
            node["llm_description"] = desc.strip()
            logger.info("Сгенерировано описание для %s (%s)", node.get("code") or "", node.get("label") or "")

        # Сохраняем прогресс после каждого батча
        OUTPUT_PATH.write_text(
            json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info("Генерация завершена. Результат: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()

