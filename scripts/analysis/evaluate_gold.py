from __future__ import annotations 

PROJECT_DIR =None 
GOLD_JSONL_PATH =r"data/gold/test_set_manual_draft.jsonl"
ONTOLOGY_PATH =r"data/ontology_grnti_with_llm.json"
PRECOMPUTED_EMBEDDINGS_PATH =r"data/ontology_grnti_embeddings.npz"
TEXTS_DIR =r"data/specifications/texts"




BI_ENCODER_MODEL ="deepvk/USER-bge-m3"

THRESHOLD =0.55 
TOP_K =10 
MAX_SEGMENT_LENGTH_FOR_CONTEXT =0 
RERANK_TOP_K =0 
CROSS_ENCODER_MODEL =None 
CONFIDENCE_AGGREGATION ="sum"
FILTER_SEGMENTS =True 


EVAL_KS =(1 ,3 ,5 ,10 ,20 )


MAX_PRED_CODES =max (EVAL_KS )


OUT_JSON_PATH =r"data/gold/eval_report.json"


import json 
import math 
import re 
import sys 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,Iterable ,List ,Optional ,Sequence ,Tuple 


def _resolve_project_dir ()->Path :
    return Path (__file__ ).resolve ().parents [2 ]


PROJECT_DIR =_resolve_project_dir ()
SRC_DIR =PROJECT_DIR /"src"
sys .path .insert (0 ,str (SRC_DIR ))

from annotation .annotator import DEFAULT_CROSS_ENCODER_MODEL ,EmbeddingAnnotator 


@dataclass (frozen =True )
class GoldItem :
    doc_id :str 
    text_path :Path 
    gold_codes :Tuple [str ,...]


def _load_json (path :Path )->dict :
    return json .loads (path .read_text (encoding ="utf-8"))


def _load_ontology_code_map (ontology_path :Path )->Dict [str ,str ]:
    data =_load_json (ontology_path )
    out :Dict [str ,str ]={}
    for n in data .get ("nodes",[]):
        nid =n .get ("id")
        code =n .get ("code")
        if isinstance (nid ,str )and isinstance (code ,str )and code .strip ():
            out [nid ]=code .strip ()
    return out 


def _is_leaf_grnti_code (code :str )->bool :
    parts =code .split (".")
    return len (parts )==3 and all (parts )and all (p .isdigit ()for p in parts )


def _to_level_code (code :str ,level :int )->Optional [str ]:
    parts =code .split (".")
    if level <=0 or level >len (parts ):
        return None 
    sub =parts [:level ]
    if not all (p .isdigit ()for p in sub ):
        return None 
    return ".".join (sub )


def _aggregate_codes_to_level (codes :Sequence [str ],level :int )->List [str ]:
    seen :set [str ]=set ()
    out :List [str ]=[]
    for c in codes :
        lc =_to_level_code (c ,level )
        if not lc or lc in seen :
            continue 
        seen .add (lc )
        out .append (lc )
    return out 


def _read_gold_items (gold_path :Path ,texts_dir :Path )->List [GoldItem ]:
    items :List [GoldItem ]=[]
    for raw_line in gold_path .read_text (encoding ="utf-8").splitlines ():
        line =raw_line .strip ()
        if not line :
            continue 
        obj =json .loads (line )
        doc_id =str (obj .get ("doc_id","")).strip ()
        if not doc_id :
            continue 

        gold_codes_raw =obj .get ("gold_codes")or obj .get ("labels")or []
        gold_codes :List [str ]=[]
        if isinstance (gold_codes_raw ,list ):

            for x in gold_codes_raw :
                if isinstance (x ,str ):
                    gold_codes .append (x .strip ())
                elif isinstance (x ,dict )and isinstance (x .get ("code"),str ):
                    gold_codes .append (x ["code"].strip ())

        gold_codes =[c for c in gold_codes if _is_leaf_grnti_code (c )]
        gold_codes =sorted (set (gold_codes ))

        tp =obj .get ("text_path")
        if isinstance (tp ,str )and tp .strip ():
            text_path =Path (tp )
        else :
            text_path =texts_dir /f"{doc_id}.txt"

        items .append (GoldItem (doc_id =doc_id ,text_path =text_path ,gold_codes =tuple (gold_codes )))
    return items 


def precision_at_k (pred :Sequence [str ],gold :Sequence [str ],k :int )->float :
    if k <=0 :
        return 0.0 
    pred_k =pred [:k ]
    if not pred_k :
        return 0.0 
    g =set (gold )
    return sum (1 for p in pred_k if p in g )/float (len (pred_k ))


def recall_at_k (pred :Sequence [str ],gold :Sequence [str ],k :int )->float :
    if k <=0 :
        return 0.0 
    g =set (gold )
    if not g :
        return 1.0 
    pred_k =pred [:k ]
    return sum (1 for p in pred_k if p in g )/float (len (g ))


def mrr_at_k (pred :Sequence [str ],gold :Sequence [str ],k :int )->float :
    g =set (gold )
    if not g :
        return 1.0 
    for i ,p in enumerate (pred [:k ],start =1 ):
        if p in g :
            return 1.0 /float (i )
    return 0.0 


def ap_at_k (pred :Sequence [str ],gold :Sequence [str ],k :int )->float :
    g =set (gold )
    if not g :
        return 1.0 
    hits =0 
    s =0.0 
    for i ,p in enumerate (pred [:k ],start =1 ):
        if p in g :
            hits +=1 
            s +=hits /float (i )
    return s /float (len (g ))


def mean (xs :Iterable [float ])->float :
    xs =list (xs )
    return sum (xs )/float (len (xs ))if xs else 0.0 


def run_predictions_for_doc (
text_path :Path ,
annotator :EmbeddingAnnotator ,
competency_id_to_code :Dict [str ,str ],
)->List [str ]:
    text =text_path .read_text (encoding ="utf-8",errors ="replace")
    anns =annotator .annotate (
    text =text ,
    threshold =THRESHOLD ,
    top_k =TOP_K ,
    max_segment_length_for_context =MAX_SEGMENT_LENGTH_FOR_CONTEXT ,
    rerank_top_k =RERANK_TOP_K ,
    confidence_aggregation =CONFIDENCE_AGGREGATION ,
    filter_segments =FILTER_SEGMENTS ,
    )
    codes :List [str ]=[]
    for ann in anns :
        cid =ann .get ("competency_id")
        if not isinstance (cid ,str ):
            continue 
        code =competency_id_to_code .get (cid )
        if not code :
            continue 
        code =code .strip ()
        if not _is_leaf_grnti_code (code ):
            continue 
        if code not in codes :
            codes .append (code )
            if len (codes )>=MAX_PRED_CODES :
                break 
    return codes 


def main ()->None :
    gold_path =PROJECT_DIR /GOLD_JSONL_PATH 
    ontology_path =PROJECT_DIR /ONTOLOGY_PATH 
    texts_dir =PROJECT_DIR /TEXTS_DIR 
    out_path =PROJECT_DIR /OUT_JSON_PATH 

    if not gold_path .exists ():
        raise FileNotFoundError (f"GOLD файл не найден: {gold_path}")
    if not ontology_path .exists ():
        raise FileNotFoundError (f"Онтология не найдена: {ontology_path}")

    emb_path =PROJECT_DIR /PRECOMPUTED_EMBEDDINGS_PATH 
    precomputed =emb_path if emb_path .exists ()else None 

    competency_id_to_code =_load_ontology_code_map (ontology_path )
    items =_read_gold_items (gold_path ,texts_dir )
    if not items :
        raise RuntimeError (f"В GOLD нет записей: {gold_path}")

    cross_encoder_model =CROSS_ENCODER_MODEL 
    if RERANK_TOP_K >0 and cross_encoder_model is None :
        cross_encoder_model =DEFAULT_CROSS_ENCODER_MODEL 
    if RERANK_TOP_K ==0 :
        cross_encoder_model =None 

    annotator =EmbeddingAnnotator (
    ontology_path =ontology_path ,
    model_name =BI_ENCODER_MODEL ,
    cross_encoder_model =cross_encoder_model ,
    precomputed_embeddings_path =precomputed ,
    )

    per_doc :List [dict ]=[]
    agg_leaf :Dict [int ,Dict [str ,List [float ]]]={k :{"p":[],"r":[],"mrr":[],"ap":[]}for k in EVAL_KS }
    agg_parent :Dict [int ,Dict [str ,List [float ]]]={k :{"p":[],"r":[],"mrr":[],"ap":[]}for k in EVAL_KS }
    agg_grand :Dict [int ,Dict [str ,List [float ]]]={k :{"p":[],"r":[],"mrr":[],"ap":[]}for k in EVAL_KS }
    docs_missing_text =0 
    docs_no_pred =0 

    try :
        from tqdm import tqdm 
        iterator =tqdm (items ,desc ="GOLD eval")
    except Exception :
        iterator =items 

    for it in iterator :
        if not it .text_path .exists ():
            docs_missing_text +=1 
            per_doc .append (
            {
            "doc_id":it .doc_id ,
            "error":f"text_path не найден: {str(it.text_path)}",
            "gold":list (it .gold_codes ),
            "pred":[],
            }
            )
            continue 

        pred_codes =run_predictions_for_doc (
        text_path =it .text_path ,
        annotator =annotator ,
        competency_id_to_code =competency_id_to_code ,
        )
        if not pred_codes :
            docs_no_pred +=1 


        doc_metrics :Dict [str ,float ]={}
        for k in EVAL_KS :
            p =precision_at_k (pred_codes ,it .gold_codes ,k )
            r =recall_at_k (pred_codes ,it .gold_codes ,k )
            mrr =mrr_at_k (pred_codes ,it .gold_codes ,k )
            ap =ap_at_k (pred_codes ,it .gold_codes ,k )
            doc_metrics [f"P@{k}"]=p 
            doc_metrics [f"R@{k}"]=r 
            doc_metrics [f"Recall@{k}"]=r 
            doc_metrics [f"MRR@{k}"]=mrr 
            doc_metrics [f"AP@{k}"]=ap 
            agg_leaf [k ]["p"].append (p )
            agg_leaf [k ]["r"].append (r )
            agg_leaf [k ]["mrr"].append (mrr )
            agg_leaf [k ]["ap"].append (ap )


        gold_parent =_aggregate_codes_to_level (it .gold_codes ,2 )
        pred_parent =_aggregate_codes_to_level (pred_codes ,2 )
        doc_metrics_parent :Dict [str ,float ]={}
        for k in EVAL_KS :
            p =precision_at_k (pred_parent ,gold_parent ,k )
            r =recall_at_k (pred_parent ,gold_parent ,k )
            mrr =mrr_at_k (pred_parent ,gold_parent ,k )
            ap =ap_at_k (pred_parent ,gold_parent ,k )
            doc_metrics_parent [f"P@{k}"]=p 
            doc_metrics_parent [f"R@{k}"]=r 
            doc_metrics_parent [f"Recall@{k}"]=r 
            doc_metrics_parent [f"MRR@{k}"]=mrr 
            doc_metrics_parent [f"AP@{k}"]=ap 
            agg_parent [k ]["p"].append (p )
            agg_parent [k ]["r"].append (r )
            agg_parent [k ]["mrr"].append (mrr )
            agg_parent [k ]["ap"].append (ap )


        gold_grand =_aggregate_codes_to_level (it .gold_codes ,1 )
        pred_grand =_aggregate_codes_to_level (pred_codes ,1 )
        doc_metrics_grand :Dict [str ,float ]={}
        for k in EVAL_KS :
            p =precision_at_k (pred_grand ,gold_grand ,k )
            r =recall_at_k (pred_grand ,gold_grand ,k )
            mrr =mrr_at_k (pred_grand ,gold_grand ,k )
            ap =ap_at_k (pred_grand ,gold_grand ,k )
            doc_metrics_grand [f"P@{k}"]=p 
            doc_metrics_grand [f"R@{k}"]=r 
            doc_metrics_grand [f"Recall@{k}"]=r 
            doc_metrics_grand [f"MRR@{k}"]=mrr 
            doc_metrics_grand [f"AP@{k}"]=ap 
            agg_grand [k ]["p"].append (p )
            agg_grand [k ]["r"].append (r )
            agg_grand [k ]["mrr"].append (mrr )
            agg_grand [k ]["ap"].append (ap )

        per_doc .append (
        {
        "doc_id":it .doc_id ,
        "text_path":str (it .text_path ),
        "gold":list (it .gold_codes ),
        "pred":pred_codes ,
        "metrics":doc_metrics ,
        "metrics_parent":doc_metrics_parent ,
        "metrics_grandparent":doc_metrics_grand ,
        }
        )

    summary ={
    "params":{
    "bi_encoder_model":BI_ENCODER_MODEL ,
    "threshold":THRESHOLD ,
    "top_k":TOP_K ,
    "max_segment_length_for_context":MAX_SEGMENT_LENGTH_FOR_CONTEXT ,
    "rerank_top_k":RERANK_TOP_K ,
    "cross_encoder_model":CROSS_ENCODER_MODEL ,
    "confidence_aggregation":CONFIDENCE_AGGREGATION ,
    "filter_segments":FILTER_SEGMENTS ,
    "eval_ks":list (EVAL_KS ),
    "max_pred_codes":MAX_PRED_CODES ,
    "gold_path":str (gold_path ),
    "ontology_path":str (ontology_path ),
    "precomputed_embeddings_path":str (precomputed )if precomputed else None ,
    },
    "stats":{
    "docs_total":len (items ),
    "docs_missing_text":docs_missing_text ,
    "docs_no_pred":docs_no_pred ,
    "avg_gold_labels":mean (len (r .get ("gold",[]))for r in per_doc if isinstance (r .get ("gold"),list )),
    "avg_pred_labels":mean (len (r .get ("pred",[]))for r in per_doc if isinstance (r .get ("pred"),list )),
    },
    "macro":{
    f"P@{k}":mean (agg_leaf [k ]["p"])for k in EVAL_KS 
    }
    |{f"R@{k}":mean (agg_leaf [k ]["r"])for k in EVAL_KS }
    |{f"Recall@{k}":mean (agg_leaf [k ]["r"])for k in EVAL_KS }
    |{f"MRR@{k}":mean (agg_leaf [k ]["mrr"])for k in EVAL_KS }
    |{f"MAP@{k}":mean (agg_leaf [k ]["ap"])for k in EVAL_KS },
    "macro_parent":{
    f"P@{k}":mean (agg_parent [k ]["p"])for k in EVAL_KS 
    }
    |{f"R@{k}":mean (agg_parent [k ]["r"])for k in EVAL_KS }
    |{f"Recall@{k}":mean (agg_parent [k ]["r"])for k in EVAL_KS }
    |{f"MRR@{k}":mean (agg_parent [k ]["mrr"])for k in EVAL_KS }
    |{f"MAP@{k}":mean (agg_parent [k ]["ap"])for k in EVAL_KS },
    "macro_grandparent":{
    f"P@{k}":mean (agg_grand [k ]["p"])for k in EVAL_KS 
    }
    |{f"R@{k}":mean (agg_grand [k ]["r"])for k in EVAL_KS }
    |{f"Recall@{k}":mean (agg_grand [k ]["r"])for k in EVAL_KS }
    |{f"MRR@{k}":mean (agg_grand [k ]["mrr"])for k in EVAL_KS }
    |{f"MAP@{k}":mean (agg_grand [k ]["ap"])for k in EVAL_KS },
    "per_doc":per_doc ,
    }

    out_path .parent .mkdir (parents =True ,exist_ok =True )
    out_path .write_text (json .dumps (summary ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")
    print (f"OK: сохранён отчёт {out_path}")
    print ("---- Summary ----")
    print (
    f"docs={summary['stats']['docs_total']}, "
    f"missing_text={summary['stats']['docs_missing_text']}, "
    f"no_pred={summary['stats']['docs_no_pred']}, "
    f"avg_gold={summary['stats']['avg_gold_labels']:.2f}, "
    f"avg_pred={summary['stats']['avg_pred_labels']:.2f}"
    )

    for k in EVAL_KS :
        p =summary ["macro"].get (f"P@{k}",0.0 )
        r =summary ["macro"].get (f"Recall@{k}",summary ["macro"].get (f"R@{k}",0.0 ))
        mrr =summary ["macro"].get (f"MRR@{k}",0.0 )
        mp =summary ["macro"].get (f"MAP@{k}",0.0 )
        print (f"P@{k}={p:.4f}  Recall@{k}={r:.4f}  MRR@{k}={mrr:.4f}  MAP@{k}={mp:.4f}")


if __name__ =="__main__":
    main ()

