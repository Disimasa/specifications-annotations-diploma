from __future__ import annotations 

import json 
import logging 
import os 
import time 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,List ,Tuple 
from urllib .parse import urljoin ,urlparse ,parse_qs 

import requests 
from bs4 import BeautifulSoup 
from tqdm import tqdm 


BASE_URL ="https://grnti.ru/"
PROJECT_DIR =Path (__file__ ).resolve ().parents [1 ]
OUTPUT_PATH =PROJECT_DIR /"data"/"ontology_grnti.json"


COMP_PREFIX ="http://example.org/competencies#"
PREDICATE_CONTAINS =f"{COMP_PREFIX}содержит"
ROOT_ID ="http://example.org/grnti_root"


logging .basicConfig (level =logging .INFO ,format ="[%(asctime)s] [%(levelname)s] %(message)s")
logger =logging .getLogger ("grnti")


@dataclass 
class GrntiNode :
    code :str 
    title :str 
    level :int 

    @property 
    def full_label (self )->str :
        return f"{self.code} {self.title}".strip ()

    @property 
    def node_id (self )->str :

        safe_code =self .code .replace (".","_")
        return f"{COMP_PREFIX}GRNTI_{safe_code}"


def fetch_html (url :str ,max_retries :int =5 ,timeout :int =30 ,base_sleep :float =1.0 )->str :
    proxy_url =os .environ .get ("GRNTI_PROXY","").strip ()or None 
    proxies ={"http":proxy_url ,"https":proxy_url }if proxy_url else None 
    headers ={
    "User-Agent":"Mozilla/5.0 (compatible; grnti-parser/1.0; +https://grnti.ru/)",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_err :Exception |None =None 
    for attempt in range (1 ,max_retries +1 ):
        try :
            resp =requests .get (url ,timeout =timeout ,headers =headers ,proxies =proxies )
            resp .raise_for_status ()
            return resp .text 
        except (requests .Timeout ,requests .ConnectionError ,requests .HTTPError )as exc :
            last_err =exc 
            if attempt ==max_retries :
                logger .error ("Не удалось получить %s после %d попыток: %s",url ,attempt ,exc )
                raise 
            sleep_for =base_sleep *attempt 
            logger .warning (
            "Ошибка при запросе %s (попытка %d/%d): %s. Повтор через %.1f c",
            url ,
            attempt ,
            max_retries ,
            exc ,
            sleep_for ,
            )
            time .sleep (sleep_for )


    raise RuntimeError (f"Не удалось получить {url}: {last_err}")


def parse_links_with_params (html :str )->List [Tuple [Dict [str ,str ],str ]]:
    soup =BeautifulSoup (html ,"html.parser")
    result :List [Tuple [Dict [str ,str ],str ]]=[]
    for a in soup .find_all ("a",href =True ):
        href =a ["href"]
        full_url =urljoin (BASE_URL ,href )
        parsed =urlparse (full_url )
        qs =parse_qs (parsed .query )
        params ={k :v [0 ]for k ,v in qs .items ()if k in {"p1","p2","p3"}and v }
        if not params :
            continue 
        text =a .get_text (strip =True )
        if not text :
            continue 
        result .append ((params ,text ))
    return result 


def parse_top_level ()->List [GrntiNode ]:
    html =fetch_html (BASE_URL )
    links =parse_links_with_params (html )
    nodes :Dict [str ,GrntiNode ]={}

    for params ,text in links :
        if "p1"not in params or len (params )!=1 :
            continue 
        code =params ["p1"]


        title =text 
        if text .startswith (code ):
            title =text [len (code ):].strip (" :–-")
        nodes [code ]=GrntiNode (code =code ,title =title ,level =1 )


    return [nodes [k ]for k in sorted (nodes .keys ())]


def parse_second_level (p1 :str )->List [GrntiNode ]:
    url =f"{BASE_URL}?p1={p1}"
    html =fetch_html (url )
    links =parse_links_with_params (html )
    nodes :Dict [str ,GrntiNode ]={}

    for params ,text in links :
        if params .get ("p1")!=p1 or "p2"not in params or len (params )!=2 :
            continue 
        p2 =params ["p2"]
        code =f"{p1}.{p2}"

        title =text 
        if text .startswith (code ):
            title =text [len (code ):].strip (" :–-")
        nodes [code ]=GrntiNode (code =code ,title =title ,level =2 )

    return [nodes [k ]for k in sorted (nodes .keys ())]


def parse_third_level (p1 :str ,p2 :str )->List [GrntiNode ]:
    url =f"{BASE_URL}?p1={p1}&p2={p2}"
    html =fetch_html (url )
    links =parse_links_with_params (html )
    nodes :Dict [str ,GrntiNode ]={}
    prefix =f"{p1}.{p2}."

    for params ,text in links :
        if params .get ("p1")!=p1 or params .get ("p2")!=p2 or "p3"not in params :
            continue 
        p3 =params ["p3"]
        code =f"{p1}.{p2}.{p3}"

        if not text .startswith (prefix ):

            continue 
        title =text [len (code ):].strip (" :–-")
        nodes [code ]=GrntiNode (code =code ,title =title ,level =3 )

    return [nodes [k ]for k in sorted (nodes .keys ())]


def build_ontology ()->Dict [str ,list ]:
    nodes :Dict [str ,GrntiNode ]={}
    links :List [Dict [str ,str ]]=[]


    ontology_nodes :List [Dict [str ,object ]]=[
    {
    "id":ROOT_ID ,
    "label":"ГРНТИ",
    "type":"class",
    "children":[],
    "version":0 ,
    "author":"grnti.ru",
    "description":"Государственный рубрикатор научно-технической информации (ГРНТИ).",
    }
    ]

    logger .info ("Парсим верхний уровень ГРНТИ…")
    top_level =parse_top_level ()
    logger .info ("Найдено %d верхнеуровневых разделов (p1)",len (top_level ))

    for sec in tqdm (top_level ,desc ="p1 (верхний уровень)",unit ="раздел"):
        nodes [sec .code ]=sec 
        ontology_nodes .append (
        {
        "id":sec .node_id ,
        "label":sec .full_label ,
        "type":"class",
        "children":[],
        "version":0 ,
        "author":"grnti.ru",
        "description":sec .full_label ,
        }
        )
        links .append (
        {
        "source":ROOT_ID ,
        "target":sec .node_id ,
        "predicate":PREDICATE_CONTAINS ,
        }
        )


        p1 =sec .code 
        subnodes =parse_second_level (p1 )
        logger .info ("p1=%s: найдено %d подразделов (p2)",p1 ,len (subnodes ))
        for sub in tqdm (subnodes ,desc =f"p2 для {p1}",leave =False ,unit ="подраздел"):
            nodes [sub .code ]=sub 
            ontology_nodes .append (
            {
            "id":sub .node_id ,
            "label":sub .full_label ,
            "type":"class",
            "children":[],
            "version":0 ,
            "author":"grnti.ru",
            "description":sub .full_label ,
            }
            )
            links .append (
            {
            "source":sec .node_id ,
            "target":sub .node_id ,
            "predicate":PREDICATE_CONTAINS ,
            }
            )


            p2 =sub .code .split (".")[1 ]
            leaf_nodes =parse_third_level (p1 ,p2 )
            if leaf_nodes :
                logger .info ("p1=%s p2=%s: найдено %d специализаций (p3)",p1 ,p2 ,len (leaf_nodes ))
            for spec in leaf_nodes :

                if spec .code not in nodes :
                    nodes [spec .code ]=spec 
                    ontology_nodes .append (
                    {
                    "id":spec .node_id ,
                    "label":spec .full_label ,
                    "type":"class",
                    "children":[],
                    "version":0 ,
                    "author":"grnti.ru",
                    "description":spec .full_label ,
                    }
                    )
                links .append (
                {
                "source":sub .node_id ,
                "target":nodes [spec .code ].node_id ,
                "predicate":PREDICATE_CONTAINS ,
                }
                )

    return {"nodes":ontology_nodes ,"links":links }


def main ()->None :
    OUTPUT_PATH .parent .mkdir (parents =True ,exist_ok =True )
    ontology =build_ontology ()
    OUTPUT_PATH .write_text (json .dumps (ontology ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")
    print (f"Сохранено {len(ontology['nodes'])} узлов и {len(ontology['links'])} связей в {OUTPUT_PATH}")


if __name__ =="__main__":
    main ()

