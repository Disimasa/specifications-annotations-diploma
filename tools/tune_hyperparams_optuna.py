from __future__ import annotations 

import csv 
import json 
import sys 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,Iterable ,List ,Optional ,Sequence ,Tuple 

import optuna 
from optuna import logging as optuna_logging 


PROJECT_DIR =Path (__file__ ).resolve ().parents [1 ]
if str (PROJECT_DIR )not in sys .path :
    sys .path .insert (0 ,str (PROJECT_DIR ))

from annotation .annotator import (
DEFAULT_CROSS_ENCODER_MODEL ,
EmbeddingAnnotator ,
)







ONTOLOGY_PATH =PROJECT_DIR /"data"/"ontology_grnti_with_llm.json"
EMBEDDINGS_PATH =PROJECT_DIR /"data"/"ontology_grnti_embeddings.npz"
GOLD_PATH =PROJECT_DIR /"data"/"gold"/"gisnauka_samples.csv"

BI_ENCODER_MODEL ="deepvk/USER-bge-m3"





CE_MODEL_HF_DITY =DEFAULT_CROSS_ENCODER_MODEL 
CE_MODEL_RUSBEIR ="models/cross-encoder-rusbeir/final_model"


CROSS_ENCODER_MODEL_MAP :Dict [str ,str ]={"hf_dity":CE_MODEL_HF_DITY }
if (PROJECT_DIR /CE_MODEL_RUSBEIR ).exists ():
    CROSS_ENCODER_MODEL_MAP ["local_rusbeir"]=CE_MODEL_RUSBEIR 

CROSS_ENCODER_MODEL_CHOICES =list (CROSS_ENCODER_MODEL_MAP .keys ())


THRESHOLD_RANGE =(0.3 ,0.9 )
TOP_K_RANGE =(5 ,30 )
MAX_SEGMENT_LENGTH_RANGE =(0 ,800 )
RERANK_TOP_K_RANGE =(0 ,30 )
CONFIDENCE_AGGREGATION_CHOICES =[
"sum",
"sum_log_count",
"mean_log_count",
]
FILTER_SEGMENTS_CHOICES =[True ,False ]
USE_CE_DOC_SCORE_CHOICES =[False ,True ]


EVAL_K =20 
MAX_PRED_CODES =EVAL_K 


N_TRIALS =100 
STUDY_NAME ="annotator_hparam_search_r20"
STORAGE_URL =None 


BEST_PARAMS_BASE_DIR =PROJECT_DIR /"data"/"gold"/"optuna_runs"







def _is_leaf_grnti_code (code :str )->bool :
    parts =code .split (".")
    return len (parts )==3 and all (p .isdigit ()for p in parts )


@dataclass 
class GoldItem :
    doc_id :str 
    text :str 
    gold_codes :Tuple [str ,...]
    top_code :Optional [str ]=None 


def _read_gold_csv (path :Path )->List [GoldItem ]:
    items :List [GoldItem ]=[]


    try :
        csv .field_size_limit (sys .maxsize )
    except (OverflowError ,ValueError ):
        csv .field_size_limit (10_000_000 )

    with path .open (encoding ="utf-8",newline ="")as f :
        reader =csv .DictReader (f )
        for i ,row in enumerate (reader ):
            title =(row .get ("title")or "").strip ()
            abstract =(row .get ("abstract")or "").strip ()
            codes_raw =(row .get ("grnti_codes")or "").strip ()
            if not codes_raw :
                continue 
            codes =[c .strip ()for c in codes_raw .split (";")if c .strip ()]
            codes =[c for c in codes if _is_leaf_grnti_code (c )]
            codes =sorted (set (codes ))
            if not codes :
                continue 
            text =f"{title}\n\n{abstract}".strip ()
            if not text :
                continue 
            doc_id =row .get ("doc_id")or f"gisnauka_{i}"
            top_code =(row .get ("top_code")or "").strip ()or codes [0 ].split (".")[0 ]
            items .append (GoldItem (doc_id =str (doc_id ),text =text ,gold_codes =tuple (codes ),top_code =top_code ))
    return items 



_SAMPLE_ITEMS :List [GoldItem ]=[]


def _get_sample_items ()->List [GoldItem ]:
    global _SAMPLE_ITEMS 
    if _SAMPLE_ITEMS :
        return _SAMPLE_ITEMS 

    all_items =_read_gold_csv (GOLD_PATH )
    if not all_items :
        raise RuntimeError (f"Нет данных в GOLD: {GOLD_PATH}")

    by_top :Dict [str ,List [GoldItem ]]={}
    for it in all_items :
        top =(it .top_code or (it .gold_codes [0 ].split (".")[0 ]if it .gold_codes else "")).strip ()
        if not top :
            continue 
        by_top .setdefault (top ,[]).append (it )

    sample :List [GoldItem ]=[]
    for top in sorted (by_top .keys ()):
        docs =by_top [top ]
        if docs :
            sample .append (docs [0 ])

    if not sample :
        raise RuntimeError ("Не удалось собрать фиксированную подвыборку документов по верхнеуровневым рубрикам.")

    _SAMPLE_ITEMS =sample 
    return _SAMPLE_ITEMS 


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


def mean (xs :Iterable [float ])->float :
    xs =list (xs )
    return sum (xs )/float (len (xs ))if xs else 0.0 


def run_predictions_for_doc (
text :str ,
annotator :EmbeddingAnnotator ,
competency_id_to_code :Dict [str ,str ],
max_pred_codes :int ,
)->List [str ]:
    anns =annotator .annotate (
    text =text ,
    threshold =run_predictions_for_doc .threshold ,
    top_k =run_predictions_for_doc .top_k ,
    max_segment_length_for_context =run_predictions_for_doc .max_segment_length ,
    rerank_top_k =run_predictions_for_doc .rerank_top_k ,
    confidence_aggregation =run_predictions_for_doc .conf_agg ,
    filter_segments =run_predictions_for_doc .filter_segments ,
    use_cross_encoder_doc_score =run_predictions_for_doc .use_ce_doc_score ,
    )
    codes :List [str ]=[]
    for ann in anns :
        cid =ann .get ("competency_id")
        if not isinstance (cid ,str ):
            continue 
        code =competency_id_to_code .get (cid )
        if not isinstance (code ,str ):
            continue 
        code =code .strip ()
        if not _is_leaf_grnti_code (code ):
            continue 
        if code not in codes :
            codes .append (code )
            if len (codes )>=max_pred_codes :
                break 
    return codes 


def build_annotator (
ontology_path :Path ,
embeddings_path :Path ,
bi_model :str ,
ce_model :Optional [str ],
)->EmbeddingAnnotator :
    precomputed =embeddings_path if embeddings_path .exists ()else None 
    annotator =EmbeddingAnnotator (
    ontology_path =ontology_path ,
    model_name =bi_model ,
    cross_encoder_model =ce_model ,
    precomputed_embeddings_path =precomputed ,
    )
    return annotator 


def objective (trial :optuna .Trial )->float :

    threshold =trial .suggest_float ("threshold",*THRESHOLD_RANGE )
    top_k =trial .suggest_int ("top_k",*TOP_K_RANGE )
    max_segment_length =trial .suggest_int ("max_segment_length_for_context",*MAX_SEGMENT_LENGTH_RANGE )
    rerank_top_k =trial .suggest_int ("rerank_top_k",*RERANK_TOP_K_RANGE )
    conf_agg =trial .suggest_categorical ("confidence_aggregation",CONFIDENCE_AGGREGATION_CHOICES )
    filter_segments =trial .suggest_categorical ("filter_segments",FILTER_SEGMENTS_CHOICES )
    use_ce_doc_score =trial .suggest_categorical ("use_ce_doc_score",USE_CE_DOC_SCORE_CHOICES )



    if rerank_top_k ==0 :

        ce_model =None 
        use_ce_doc_score =False 
    else :

        ce_model_key =trial .suggest_categorical ("cross_encoder_model",CROSS_ENCODER_MODEL_CHOICES )
        ce_model =CROSS_ENCODER_MODEL_MAP [ce_model_key ]


    sample =_get_sample_items ()


    annotator =build_annotator (
    ontology_path =ONTOLOGY_PATH ,
    embeddings_path =EMBEDDINGS_PATH ,
    bi_model =BI_ENCODER_MODEL ,
    ce_model =ce_model ,
    )


    onto =json .loads (ONTOLOGY_PATH .read_text (encoding ="utf-8"))
    competency_id_to_code :Dict [str ,str ]={}
    for n in onto .get ("nodes",[]):
        nid =n .get ("id")
        code =n .get ("code")
        if isinstance (nid ,str )and isinstance (code ,str )and code .strip ():
            competency_id_to_code [nid ]=code .strip ()


    run_predictions_for_doc .threshold =threshold 
    run_predictions_for_doc .top_k =top_k 
    run_predictions_for_doc .max_segment_length =max_segment_length 
    run_predictions_for_doc .rerank_top_k =rerank_top_k 
    run_predictions_for_doc .conf_agg =conf_agg 
    run_predictions_for_doc .filter_segments =filter_segments 
    run_predictions_for_doc .use_ce_doc_score =use_ce_doc_score 

    recalls :List [float ]=[]

    for it in sample :
        pred_codes =run_predictions_for_doc (
        text =it .text ,
        annotator =annotator ,
        competency_id_to_code =competency_id_to_code ,
        max_pred_codes =MAX_PRED_CODES ,
        )
        r20 =recall_at_k (pred_codes ,it .gold_codes ,EVAL_K )
        recalls .append (r20 )

    macro_r20 =mean (recalls )
    return macro_r20 


def main ()->None :
    if not ONTOLOGY_PATH .exists ():
        raise FileNotFoundError (f"Онтология не найдена: {ONTOLOGY_PATH}")
    if not EMBEDDINGS_PATH .exists ():
        raise FileNotFoundError (f"Эмбеддинги онтологии не найдены: {EMBEDDINGS_PATH}")
    if not GOLD_PATH .exists ():
        raise FileNotFoundError (f"GOLD CSV не найден: {GOLD_PATH}")


    optuna_logging .set_verbosity (optuna_logging .INFO )

    sampler =optuna .samplers .TPESampler ()

    if STORAGE_URL :
        study =optuna .create_study (
        study_name =STUDY_NAME ,
        storage =STORAGE_URL ,
        direction ="maximize",
        sampler =sampler ,
        load_if_exists =True ,
        )
    else :
        study =optuna .create_study (direction ="maximize",sampler =sampler )


    from datetime import datetime 

    run_dir =BEST_PARAMS_BASE_DIR /datetime .now ().strftime ("%Y%m%d_%H%M%S")
    run_dir .mkdir (parents =True ,exist_ok =True )
    best_params_path =run_dir /"best_params.json"


    initial_params ={
    "threshold":0.55 ,
    "top_k":10 ,
    "max_segment_length_for_context":0 ,
    "rerank_top_k":0 ,
    "confidence_aggregation":"sum",
    "filter_segments":True ,
    "use_ce_doc_score":False ,
    "cross_encoder_model":CROSS_ENCODER_MODEL_CHOICES [0 ],
    }
    study .enqueue_trial (initial_params )

    def log_trial (study :optuna .study .Study ,trial :optuna .trial .FrozenTrial )->None :
        logger =optuna_logging .get_logger ("optuna")
        best =study .best_trial 
        logger .info (
        "Trial %d finished with value=%.4f, params=%s. Best so far: trial %d with value=%.4f.",
        trial .number ,
        trial .value ,
        trial .params ,
        best .number ,
        best .value ,
        )

        best_payload ={
        "value":best .value ,
        "params":best .params ,
        "number":best .number ,
        }
        best_params_path .write_text (json .dumps (best_payload ,ensure_ascii =False ,indent =2 ),encoding ="utf-8")

    study .optimize (objective ,n_trials =N_TRIALS ,callbacks =[log_trial ])

    print ("Лучший trial:")
    print (f"  value (Recall@{EVAL_K}): {study.best_value:.4f}")
    print ("  params:")
    for k ,v in study .best_params .items ():
        print (f"    {k}: {v}")


if __name__ =="__main__":
    main ()

