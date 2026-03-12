from __future__ import annotations 

import json 
import logging 
import os 
import re 
import time 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Tuple 

import requests 
from dotenv import load_dotenv 
from tqdm import tqdm 

PROJECT_DIR =Path (__file__ ).resolve ().parents [2 ]
import sys 
SRC_DIR =PROJECT_DIR /"src"
sys .path .insert (0 ,str (SRC_DIR ))

from annotation .segmenter import TextSegmenter 
from annotation .segment_filter import SegmentFilter 

TEXTS_DIR =PROJECT_DIR /"data"/"specifications"/"texts"
ONTOLOGY_PATH =PROJECT_DIR /"data"/"ontology_grnti_with_llm.json"
OUT_PATH =PROJECT_DIR /"data"/"gold"/"test_set_manual_draft.jsonl"

OPENROUTER_API_URL ="https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL ="qwen/qwen3-vl-235b-a22b-thinking"

MAX_SEGMENTS_FOR_LLM =60 


LLM_TIMEOUT =120 
LLM_MAX_RETRIES =5 
LLM_BASE_SLEEP =2.0 

logging .basicConfig (level =logging .INFO ,format ="[%(asctime)s] [%(levelname)s] %(message)s")
logger =logging .getLogger ("gold_gen")


def load_api_key ()->str :
    load_dotenv ()
    api_key =os .getenv ("OPENROUTER_API_KEY","").strip ()
    if not api_key :
        raise RuntimeError ("OPENROUTER_API_KEY не найден в .env")
    return api_key 


def load_ontology (ontology_path :Path )->Tuple [Dict [str ,dict ],Dict [str ,str ],Dict [str ,str ]]:
    data =json .loads (ontology_path .read_text (encoding ="utf-8"))
    nodes =data .get ("nodes",[])
    nodes_by_id :Dict [str ,dict ]={}
    code_by_id :Dict [str ,str ]={}
    id_by_code :Dict [str ,str ]={}

    for n in nodes :
        nid =n .get ("id")
        if not isinstance (nid ,str ):
            continue 
        nodes_by_id [nid ]=n 
        code =n .get ("code")
        if isinstance (code ,str )and code .strip ():
            code =code .strip ()
            code_by_id [nid ]=code 
            id_by_code [code ]=nid 

    return nodes_by_id ,code_by_id ,id_by_code 


def is_leaf_code (code :str )->bool :

    parts =code .split (".")
    return len (parts )==3 and all (parts )

def segment_text (text :str )->List [str ]:
    seg =TextSegmenter ()
    filt =SegmentFilter ()
    segments =seg .segment (text )
    segments =filt .filter_segments (segments )

    out :List [str ]=[]
    for s in segments :
        s =s .strip ()
        if not s :
            continue 
        if len (s )>600 :
            s =s [:600 ]+"…"
        out .append (s )
    return out 


def chunk_list (items :List [Tuple [str ,str ]],size :int )->List [List [Tuple [str ,str ]]]:
    return [items [i :i +size ]for i in range (0 ,len (items ),size )]


def call_openrouter_json (
api_key :str ,
system_prompt :str ,
user_prompt :str ,
timeout :int =LLM_TIMEOUT ,
max_retries :int =LLM_MAX_RETRIES ,
base_sleep :float =LLM_BASE_SLEEP ,
)->dict :
    headers ={
    "Authorization":f"Bearer {api_key}",
    "Content-Type":"application/json",
    "HTTP-Referer":"https://grnti-gold-generator",
    "X-Title":"grnti-gold-generator",
    }
    payload :Dict [str ,Any ]={
    "model":OPENROUTER_MODEL ,
    "messages":[
    {"role":"system","content":system_prompt },
    {"role":"user","content":user_prompt },
    ],
    "temperature":0.2 ,
    "max_tokens":1200 ,
    }

    last_err :Optional [Exception ]=None 
    for attempt in range (1 ,max_retries +1 ):
        try :
            resp =requests .post (OPENROUTER_API_URL ,headers =headers ,json =payload ,timeout =timeout )
            if resp .status_code ==429 :
                raise RuntimeError (f"429 Too Many Requests: {resp.text}")
            resp .raise_for_status ()
            data =resp .json ()
            content =data ["choices"][0 ]["message"]["content"]
            text =content if isinstance (content ,str )else str (content )

            m =re .search (r"\{[\s\S]*\}",text )
            if not m :
                raise RuntimeError (f"Не найден JSON в ответе: {text[:300]}")
            return json .loads (m .group (0 ))
        except Exception as exc :
            last_err =exc 
            if attempt ==max_retries :
                raise 
            sleep_for =base_sleep *attempt 
            logger .warning ("LLM ошибка (%d/%d): %s. Повтор через %.1fс",attempt ,max_retries ,exc ,sleep_for )
            time .sleep (sleep_for )

    raise RuntimeError (f"LLM вызов не удался: {last_err}")


def load_existing (path :Path )->Dict [str ,dict ]:
    if not path .exists ():
        return {}
    out :Dict [str ,dict ]={}
    for line in path .read_text (encoding ="utf-8").splitlines ():
        line =line .strip ()
        if not line :
            continue 
        obj =json .loads (line )
        doc_id =obj .get ("doc_id")
        if isinstance (doc_id ,str ):
            out [doc_id ]=obj 
    return out 


def main ()->None :
    api_key =load_api_key ()

    if not TEXTS_DIR .exists ():
        raise FileNotFoundError (f"Не найдена папка: {TEXTS_DIR}")

    OUT_PATH .parent .mkdir (parents =True ,exist_ok =True )
    existing =load_existing (OUT_PATH )

    files =sorted (TEXTS_DIR .glob ("*.txt"))
    if not files :
        raise RuntimeError (f"Нет .txt в {TEXTS_DIR}")

    rows :List [dict ]=list (existing .values ())
    done_ids =set (existing .keys ())

    system_prompt =(
    "Ты помогаешь разметить документ компетенциями ГРНТИ. "
    "Твоя цель — минимальный, но достаточный набор листовых специализаций, которые реально требуются в документе. "
    "Не добавляй лишнее. Если сомневаешься — не добавляй."
    )

    for path in tqdm (files ,desc ="Документы"):
        doc_id =path .stem 
        if doc_id in done_ids :
            continue 

        text =path .read_text (encoding ="utf-8",errors ="replace")
        segments =segment_text (text )
        segments_for_llm =segments [:MAX_SEGMENTS_FOR_LLM ]

        seg_block ="\n".join (f"{i+1}) {s}"for i ,s in enumerate (segments_for_llm ))

        user_prompt =f"""
ДОКУМЕНТ (сегменты, пронумерованы):
{seg_block}

Задача:
1) Дай МИНИМАЛЬНОЕ множество ЛИСТОВЫХ специализаций ГРНТИ (коды формата XX.YY.ZZ), которые реально требуются в документе.
2) Для каждой выбранной специализации приведи 1-2 номера сегментов из документа как evidence.
3) Не выбирай специализации "на всякий случай". Если сомневаешься — не добавляй.

Ответ верни СТРОГО в JSON:
{{
  "selected": [
    {{
      "code": "XX.YY.ZZ",
      "evidence_segment_numbers": [1, 5]
    }}
  ]
}}
""".strip ()

        resp =call_openrouter_json (api_key ,system_prompt ,user_prompt )
        selected =resp .get ("selected")or []

        gold_codes :List [str ]=[]
        evidence :Dict [str ,List [str ]]={}
        for item in selected :
            code =str (item .get ("code","")).strip ()
            if not is_leaf_code (code ):
                continue 
            nums =item .get ("evidence_segment_numbers")or []
            segs :List [str ]=[]
            for n in nums :
                try :
                    idx =int (n )-1 
                    if 0 <=idx <len (segments_for_llm ):
                        segs .append (segments_for_llm [idx ])
                except Exception :
                    continue 
            if code not in gold_codes :
                gold_codes .append (code )
            if segs :
                evidence [code ]=segs [:2 ]

        row ={
        "doc_id":doc_id ,
        "text_path":str (path ),
        "gold_codes":gold_codes ,
        "evidence":evidence ,
        "meta":{
        "kind":"manual_draft_llm_minimal",
        "model":OPENROUTER_MODEL ,
        "segments_shown":len (segments_for_llm ),
        },
        }

        rows .append (row )
        done_ids .add (doc_id )


        OUT_PATH .write_text (
        "\n".join (json .dumps (r ,ensure_ascii =False )for r in rows )+"\n",
        encoding ="utf-8",
        )

    print (f"Готово: {OUT_PATH} (документов: {len(rows)})")


if __name__ =="__main__":
    main ()

