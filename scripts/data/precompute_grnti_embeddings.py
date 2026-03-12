from __future__ import annotations 

import json 
from pathlib import Path 
from typing import List 

import numpy as np 
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm 


PROJECT_DIR =Path (__file__ ).resolve ().parents [2 ]
INPUT_PATH =PROJECT_DIR /"data"/"ontology_grnti_with_llm.json"
OUTPUT_PATH =PROJECT_DIR /"data"/"ontology_grnti_embeddings.npz"

COMP_PREFIX ="http://example.org/competencies#"


DEFAULT_MODEL_NAME ="deepvk/USER-bge-m3"


def build_text_for_embedding (node :dict )->str :
    parts :List [str ]=[]
    llm_desc =(node .get ("llm_description")or "").strip ()
    full_label =(node .get ("full_label")or "").strip ()

    if full_label :
        parts .append (full_label )
    if llm_desc :
        parts .append (llm_desc )

    return ". ".join (parts ).strip (". ").strip ()


def main ()->None :
    if not INPUT_PATH .exists ():
        raise FileNotFoundError (f"Не найден входной файл {INPUT_PATH}")

    data =json .loads (INPUT_PATH .read_text (encoding ="utf-8"))
    nodes =data .get ("nodes",[])


    comp_nodes =[n for n in nodes if str (n .get ("id","")).startswith (COMP_PREFIX )]

    ids :List [str ]=[]
    texts :List [str ]=[]

    for node in comp_nodes :
        node_id =node ["id"]
        text =build_text_for_embedding (node )
        if not text :

            continue 
        ids .append (node_id )
        texts .append (text )

    if not ids :
        raise RuntimeError ("Не найдено ни одной компетенции с непустым текстом для эмбеддинга.")

    print (f"Всего компетенций для эмбеддингов: {len(ids)}")

    model =SentenceTransformer (DEFAULT_MODEL_NAME )

    embeddings =model .encode (
    texts ,
    convert_to_numpy =True ,
    normalize_embeddings =True ,
    show_progress_bar =True ,
    batch_size =64 ,
    ).astype (np .float32 )

    OUTPUT_PATH .parent .mkdir (parents =True ,exist_ok =True )
    np .savez_compressed (
    OUTPUT_PATH ,
    ids =np .array (ids ,dtype =object ),
    embeddings =embeddings ,
    )
    print (f"Сохранено {len(ids)} эмбеддингов размерности {embeddings.shape[1]} в {OUTPUT_PATH}")


if __name__ =="__main__":
    main ()

