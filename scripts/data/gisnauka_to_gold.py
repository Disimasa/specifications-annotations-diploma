from __future__ import annotations 

import csv 
import json 
import re 
import time 
from pathlib import Path 
from typing import Any ,Dict ,Iterable ,List ,Optional ,Sequence ,Tuple 

import requests 

PROJECT_DIR =Path (__file__ ).resolve ().parents [2 ]
ONTOLOGY_PATH =PROJECT_DIR /"data"/"ontology_grnti_with_llm.json"
OUT_PATH =PROJECT_DIR /"data"/"gold"/"gisnauka_samples.csv"

GISNAUKA_API_URL ="https://gisnauka.ru/api/egisu/base/search"
REQUEST_TIMEOUT =20 
SLEEP_BETWEEN_REQUESTS =1.5 

MAX_SAMPLES_PER_TOP =10 
MAX_PAGES_PER_QUERY =10 
PAGE_SIZE =10 

SESSION_HEADERS ={
"Accept":"application/json, text/plain, */*",
"Content-Type":"application/json;charset=UTF-8",
"Origin":"https://gisnauka.ru",
"Referer":"https://gisnauka.ru/global-search",
"User-Agent":(
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
"AppleWebKit/537.36 (KHTML, like Gecko) "
"Chrome/122.0.0.0 Safari/537.36"
),
}


def _is_leaf_grnti_code (code :str )->bool :
    parts =code .split (".")
    return len (parts )==3 and all (p .isdigit ()for p in parts )


def _load_top_level_grnti (ontology_path :Path )->List [Tuple [str ,str ]]:
    data =json .loads (ontology_path .read_text (encoding ="utf-8"))
    nodes =data .get ("nodes",[])
    out :List [Tuple [str ,str ]]=[]
    seen :set [str ]=set ()
    for n in nodes :
        code =n .get ("code")
        label =n .get ("label")
        if not isinstance (code ,str )or not isinstance (label ,str ):
            continue 
        code =code .strip ()
        label =label .strip ()
        if not code or not label :
            continue 
        if re .fullmatch (r"\d{2}",code )and code not in seen :
            seen .add (code )
            out .append ((code ,label ))
    return sorted (out ,key =lambda x :x [0 ])


def _extract_codes_from_item (hit :Dict [str ,Any ])->List [str ]:
    codes :List [str ]=[]
    src =hit .get ("_source",{})
    rubrics =src .get ("rubrics")or []
    if isinstance (rubrics ,list ):
        for r in rubrics :
            if not isinstance (r ,dict ):
                continue 
            code =r .get ("code")
            if isinstance (code ,str ):
                code =code .strip ()
                if _is_leaf_grnti_code (code )and code not in codes :
                    codes .append (code )
    return codes 


def _extract_abstract_from_item (hit :Dict [str ,Any ])->str :
    src =hit .get ("_source",{})
    abstract =src .get ("abstract")or src .get ("nioktr",{}).get ("annotation")or ""
    return abstract .strip ()


def _extract_title_from_item (hit :Dict [str ,Any ])->str :
    src =hit .get ("_source",{})
    title =src .get ("name")or ""
    return title .strip ()


def _search_gisnauka (query :str ,page :int )->List [Dict [str ,Any ]]:
    payload ={
    "search_query":query ,
    "critical_technologies":[],
    "dissertations":True ,
    "full_text_available":False ,
    "ikrbses":True ,
    "nioktrs":True ,
    "organization":[],
    "page":page ,
    "priority_directions":[],
    "rids":True ,
    "rubrics":[],
    "search_area":"Во всех полях",
    "sort_by":"Дата регистрации",
    "open_license":False ,
    "free_licenses":False ,
    "expert_estimation_exist":False ,
    }
    resp =requests .post (
    GISNAUKA_API_URL ,
    headers =SESSION_HEADERS ,
    json =payload ,
    timeout =REQUEST_TIMEOUT ,
    )
    resp .raise_for_status ()
    data =resp .json ()

    if not isinstance (data ,dict ):
        return []
    hits =data .get ("hits",{})
    if isinstance (hits ,dict ):
        inner =hits .get ("hits")
        if isinstance (inner ,list ):
            return inner 
    return []


def collect_samples_for_top (
top_code :str ,
top_label :str ,
max_samples :int ,
)->List [Dict [str ,Any ]]:
    samples :List [Dict [str ,Any ]]=[]
    page =1 
    while len (samples )<max_samples and page <=MAX_PAGES_PER_QUERY :
        results =_search_gisnauka (top_label ,page =page )
        if not results :
            break 
        for hit in results :
            title =_extract_title_from_item (hit )
            abstract =_extract_abstract_from_item (hit )
            codes =_extract_codes_from_item (hit )
            codes =[c for c in codes if _is_leaf_grnti_code (c )]
            if not title or not abstract :
                continue 
            if len (codes )<2 :
                continue 
            samples .append (
            {
            "source":"gisnauka",
            "top_code":top_code ,
            "top_label":top_label ,
            "title":title ,
            "abstract":abstract ,
            "grnti_codes":sorted (set (codes )),
            "raw":hit ,
            }
            )
            if len (samples )>=max_samples :
                break 
        page +=1 
        time .sleep (SLEEP_BETWEEN_REQUESTS )
    return samples 


def main ()->None :
    if not ONTOLOGY_PATH .exists ():
        raise FileNotFoundError (f"Онтология не найдена: {ONTOLOGY_PATH}")

    top_levels =_load_top_level_grnti (ONTOLOGY_PATH )
    if not top_levels :
        raise RuntimeError ("Не удалось найти верхнеуровневые разделы ГРНТИ в онтологии.")

    OUT_PATH .parent .mkdir (parents =True ,exist_ok =True )

    fieldnames =["top_code","top_label","title","abstract","grnti_codes"]
    total =0 

    with OUT_PATH .open ("w",encoding ="utf-8",newline ="")as f :
        writer =csv .DictWriter (f ,fieldnames =fieldnames )
        writer .writeheader ()

        for code ,label in top_levels :
            print (f"== Раздел {code} :: {label} ==")
            samples =collect_samples_for_top (code ,label ,MAX_SAMPLES_PER_TOP )
            print (f"  собрано сэмплов: {len(samples)}")
            for s in samples :
                writer .writerow (
                {
                "top_code":s ["top_code"],
                "top_label":s ["top_label"],
                "title":s ["title"],
                "abstract":s ["abstract"],
                "grnti_codes":";".join (s ["grnti_codes"]),
                }
                )
            total +=len (samples )

    print (f"Готово. Всего сэмплов: {total}")
    print (f"Файл: {OUT_PATH}")


if __name__ =="__main__":
    main ()

