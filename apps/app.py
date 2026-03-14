from __future__ import annotations 

import json 
import sys 
from pathlib import Path 
from tempfile import NamedTemporaryFile 
from typing import Any ,Dict ,List ,Optional 

import streamlit as st 


PROJECT_DIR =Path (__file__ ).resolve ().parents [1 ]
DEFAULT_ONTOLOGY_PATH =PROJECT_DIR /"data"/"ontology_grnti_with_llm.json"
DEFAULT_EMBEDDINGS_PATH =PROJECT_DIR /"data"/"ontology_grnti_embeddings.npz"
DEFAULT_TEXTS_DIR =PROJECT_DIR /"data"/"specifications"/"texts"

BEST_MODEL_BASE =PROJECT_DIR /"models"/"bi-encoder-gisnauka-trainer"/"best"
FALLBACK_BI_ENCODER ="deepvk/USER-bge-m3"


SRC_DIR =PROJECT_DIR /"src"
if str (SRC_DIR )not in sys .path :
    sys .path .insert (0 ,str (SRC_DIR ))

from annotation .annotator import DEFAULT_CROSS_ENCODER_MODEL ,EmbeddingAnnotator 


def _normalize_model_path (value :str )->str :
    raw =(value or "").strip ()
    if not raw :
        return raw 
    p =Path (raw )
    if p .is_absolute ():
        return str (p )

    candidate =(PROJECT_DIR /p ).resolve ()
    if candidate .exists ():
        return str (candidate )

    return raw 


def _default_embeddings_path_for_model (model_name :str )->Path :
    name =Path (model_name ).name or str (model_name )
    safe =name .replace (" ","_").replace ("\\","_").replace ("/","_")
    return DEFAULT_EMBEDDINGS_PATH .with_name (f"ontology_grnti_embeddings_{safe}.npz")


def _list_bi_encoder_options ()->List [str ]:
    options :List [str ]=[]
    # Первая опция — best обученная модель, если она есть
    if BEST_MODEL_BASE .exists ()and BEST_MODEL_BASE .is_dir ():
        subdirs =[p for p in sorted (BEST_MODEL_BASE .iterdir ())if p .is_dir ()]
        if subdirs :
            options .append (str (subdirs [0 ]))
    if not options :
        options .append (FALLBACK_BI_ENCODER )
    else :
        options .append (FALLBACK_BI_ENCODER )

    models_dir =PROJECT_DIR /"models"
    if models_dir .exists ():
        for p in sorted (models_dir .iterdir ()):
            if p .is_dir ()and p .resolve ()!=BEST_MODEL_BASE .parent .resolve ():
                options .append (str (p ))
    seen :set [str ]=set ()
    uniq :List [str ]=[]
    for o in options :
        if o in seen :
            continue 
        seen .add (o )
        uniq .append (o )
    return uniq 


def _safe_read_uploaded_text (uploaded_file )->str :
    suffix =Path (uploaded_file .name ).suffix .lower ()

    if suffix ==".txt":
        return uploaded_file .getvalue ().decode ("utf-8",errors ="replace")

    if suffix ==".docx":
        try :
            from lib .docx_to_txt import extract_text 
        except Exception as exc :
            raise RuntimeError (
            "Для загрузки .docx нужен модуль конвертации (aspose.words). "
            "Либо установите зависимости для docx, либо загрузите .txt."
            )from exc 

        with NamedTemporaryFile (suffix =".docx",delete =True )as tmp :
            tmp .write (uploaded_file .getvalue ())
            tmp .flush ()
            return extract_text (Path (tmp .name ))

    raise ValueError ("Поддерживаются только .txt и .docx")


@st .cache_data (show_spinner =False )
def _load_full_label_map (ontology_path :str )->Dict [str ,str ]:
    path =Path (ontology_path )
    if not path .exists ():
        return {}
    data =json .loads (path .read_text (encoding ="utf-8"))
    result :Dict [str ,str ]={}
    for node in data .get ("nodes",[]):
        cid =node .get ("id")
        if not isinstance (cid ,str ):
            continue 
        full_label =(node .get ("full_label")or node .get ("label")or "").strip ()
        if full_label :
            result [cid ]=full_label 
    return result 


@st .cache_resource (show_spinner =False )
def _get_annotator (
ontology_path :str ,
bi_encoder_model :str ,
cross_encoder_model :Optional [str ],
)->EmbeddingAnnotator :
    candidate =_default_embeddings_path_for_model (bi_encoder_model )
    emb_path =candidate if candidate .exists ()else DEFAULT_EMBEDDINGS_PATH 
    return EmbeddingAnnotator (
    ontology_path =Path (ontology_path ),
    model_name =bi_encoder_model ,
    cross_encoder_model =_normalize_model_path (cross_encoder_model or "")if cross_encoder_model else None ,
    precomputed_embeddings_path =emb_path if emb_path .exists ()else None ,
    )


def _build_competencies_table (
annotations :List [dict ],
top_n :int ,
full_label_map :Dict [str ,str ],
)->List [Dict [str ,Any ]]:
    rows :List [Dict [str ,Any ]]=[]
    for i ,ann in enumerate (annotations [:top_n ],start =1 ):
        matches =ann .get ("matches",[])or []
        cid =ann .get ("competency_id")
        if isinstance (cid ,str )and cid in full_label_map :
            comp_name =full_label_map [cid ]
        else :
            comp_name =ann .get ("competency_full_label",ann .get ("competency_label",""))

        comp_name_clean =comp_name .replace ("Раздел:","").replace ("Область:","")

        parts =[p .strip (" .")for p in comp_name_clean .split (".")if p .strip (" .")]
        if len (parts )>1 :
            display_name ="\n".join (parts )
        else :
            display_name =comp_name_clean .strip ()
        rows .append (
        {
        "rank":i ,
        "competency":display_name ,
        "score":float (ann .get ("doc_score",ann .get ("sum_score",ann .get ("max_confidence",0.0 )))),


        "matches":int (ann .get ("segment_count",len (matches ))),
        }
        )
    return rows 


def main ()->None :
    st .set_page_config (page_title ="Аннотация ТЗ → top‑N компетенций",layout ="wide")
    st .title ("Семантическая аннотация ТЗ по онтологии компетенций")

    with st .sidebar :
        st .header ("Параметры")

        threshold =st .slider ("Порог score",0.0 ,1.0 ,0.55 ,0.01 )
        top_k =st .number_input ("Top-K сегментов на компетенцию",min_value =1 ,max_value =100 ,value =10 ,step =1 )
        top_n =st .number_input ("Top-N компетенций (вывод)",min_value =1 ,max_value =200 ,value =20 ,step =1 )

        enable_rerank =st .checkbox (
        "Включить re-ranking (cross-encoder)",
        value =False ,
        help ="Если выключено, используется только bi-encoder.",
        )
        rerank_top_k =st .number_input (
        "Rerank: сколько компетенций переранжировать",
        min_value =0 ,
        max_value =200 ,
        value =20 if enable_rerank else 0 ,
        step =1 ,
        disabled =not enable_rerank ,
        help ="0 = re-ranking отключён. Иначе cross-encoder применяется к top-K компетенциям.",
        )

        max_segment_length_for_context =st .number_input (
        "Контекст для коротких сегментов (макс. длина, 0=выкл)",
        min_value =0 ,
        max_value =2000 ,
        value =0 ,
        step =10 ,
        help ="Если сегмент короче этого значения, добавляется контекст из соседних сегментов.",
        )

        confidence_aggregation =st .selectbox (
        "Агрегация скоров сегментов в скор документа",
        options =[
        "sum",
        "sum_log_count",
        "mean_log_count",
        "max",
        "mean",
        "median",
        "weighted_mean",
        ],
        index =0 ,
        help =(
        "sum — сумма скоров по сегментам (рекомендуется); "
        "sum_log_count — sum * log(1+кол-во сегментов); "
        "mean_log_count — mean * log(1+кол-во сегментов); "
        "остальные — max/mean/median/взвешенное среднее."
        ),
        )

        filter_segments =st .checkbox ("Фильтровать неинформативные сегменты",value =True )

        st .divider ()
        st .header ("Модели")

        bi_encoder_options =_list_bi_encoder_options ()
        bi_encoder_model =st .selectbox (
        "Bi-encoder (SentenceTransformer)",
        options =bi_encoder_options ,
        index =0 ,
        help ="Можно выбрать или указать любую модель, совместимую с sentence-transformers.",
        )

        default_cross_encoder_path =_normalize_model_path (DEFAULT_CROSS_ENCODER_MODEL )
        cross_encoder_model =st .text_input (
        "Cross-encoder (путь или HF id)",
        value =default_cross_encoder_path ,
        disabled =not enable_rerank ,
        help ="По умолчанию используется локальная модель из models/…",
        )

        st .caption (f"Онтология: `{DEFAULT_ONTOLOGY_PATH}`")

        ontology_path =str (DEFAULT_ONTOLOGY_PATH )

    col_left ,col_right =st .columns ([1 ,1 ],vertical_alignment ="top")

    with col_left :
        st .subheader ("Загрузка файла")

        uploaded =st .file_uploader ("Загрузите ТЗ (.txt или .docx)",type =["txt","docx"])

        st .caption ("Либо выберите пример из `data/specifications/texts/`.")
        examples =[]
        if DEFAULT_TEXTS_DIR .exists ():
            examples =sorted ([p .name for p in DEFAULT_TEXTS_DIR .glob ("*.txt")])
        example_name =st .selectbox ("Пример",options =["(не выбран)"]+examples )

        run_btn =st .button ("Аннотировать",type ="primary")

    text :Optional [str ]=None 
    source_label :str =""
    annotations :Optional [List [dict ]]=None 

    if run_btn :
        try :
            if uploaded is not None :
                text =_safe_read_uploaded_text (uploaded )
                source_label =uploaded .name 
            elif example_name !="(не выбран)":
                example_path =DEFAULT_TEXTS_DIR /example_name 
                text =example_path .read_text (encoding ="utf-8",errors ="replace")
                source_label =example_name 
            else :
                st .warning ("Загрузите файл или выберите пример.")
        except Exception as exc :
            st .error (str (exc ))
            text =None 

    if text is not None :
        with col_right :
            st .subheader ("Результаты")
            st .caption (f"Источник: **{source_label}**")

            try :
                annotator =_get_annotator (
                ontology_path =ontology_path ,
                bi_encoder_model =bi_encoder_model ,
                cross_encoder_model =cross_encoder_model if (enable_rerank and rerank_top_k >0 )else None ,
                )

                status_placeholder =st .empty ()
                progress_bar =st .progress (0 )

                def progress_callback (stage :str ,frac :float )->None :
                    frac_clamped =max (0.0 ,min (1.0 ,float (frac )))
                    status_placeholder .text (f"{stage} — {int(frac_clamped * 100)}%")
                    progress_bar .progress (frac_clamped )

                with st .spinner ("Считаю эмбеддинги и выполняю аннотацию…"):
                    annotations =annotator .annotate (
                    text =text ,
                    threshold =float (threshold ),
                    top_k =int (top_k ),
                    max_segment_length_for_context =int (max_segment_length_for_context ),
                    rerank_top_k =int (rerank_top_k )if enable_rerank else 0 ,
                    confidence_aggregation =str (confidence_aggregation ),
                    filter_segments =bool (filter_segments ),
                    progress_callback =progress_callback ,
                    )

            except Exception as exc :
                st .error (f"Ошибка аннотации: {exc}")


            progress_bar .empty ()
            status_placeholder .empty ()


    if text is not None and annotations :
        full_label_map =_load_full_label_map (ontology_path )

        st .divider ()
        st .subheader ("Таблица компетенций")

        table_rows =_build_competencies_table (annotations ,int (top_n ),full_label_map )
        st .dataframe (
        table_rows ,
        use_container_width =True ,
        hide_index =True ,
        column_config ={
        "rank":st .column_config .NumberColumn ("№",width ="small"),
        "competency":st .column_config .TextColumn ("Компетенция (полный путь)"),
        "score":st .column_config .NumberColumn ("Score",format ="%.4f",width ="small"),
        "matches":st .column_config .NumberColumn ("Совпадений (сегментов)",width ="small"),
        },
        )

        st .download_button (
        "Скачать результат (JSON)",
        data =json .dumps (
        {
        "source":source_label ,
        "params":{
        "ontology_path":ontology_path ,
        "bi_encoder_model":bi_encoder_model ,
        "cross_encoder_model":cross_encoder_model if (enable_rerank and rerank_top_k >0 )else None ,
        "threshold":float (threshold ),
        "top_k":int (top_k ),
        "top_n":int (top_n ),
        "rerank_top_k":int (rerank_top_k )if enable_rerank else 0 ,
        "max_segment_length_for_context":int (max_segment_length_for_context ),
        "confidence_aggregation":str (confidence_aggregation ),
        "filter_segments":bool (filter_segments ),
        },
        "annotations":annotations [:int (top_n )],
        },
        ensure_ascii =False ,
        indent =2 ,
        ).encode ("utf-8"),
        file_name ="annotations.json",
        mime ="application/json",
        use_container_width =True ,
        )


        st .divider ()
        st .subheader ("Агрегированные разделы и области")


        ontology_data =json .loads (Path (ontology_path ).read_text (encoding ="utf-8"))
        code_map :Dict [str ,str ]={n ["id"]:n .get ("code","")for n in ontology_data .get ("nodes",[])if "id"in n }
        parent_map :Dict [str ,str ]={}
        for link in ontology_data .get ("links",[]):
            src =link .get ("source")
            tgt =link .get ("target")
            if isinstance (src ,str )and isinstance (tgt ,str )and tgt not in parent_map :
                parent_map [tgt ]=src 

        def _level (node_id :str )->int :
            code =code_map .get (node_id ,"")
            if not code :
                return 0 
            return len (code .split ("."))

        level1_stats :Dict [str ,Dict [str ,Any ]]={}
        level2_stats :Dict [str ,Dict [str ,Any ]]={}

        def _accumulate (stats :Dict [str ,Dict [str ,Any ]],node_id :str ,ann :dict )->None :
            s =stats .setdefault (
            node_id ,
            {
            "competency_id":node_id ,
            "doc_score":0.0 ,
            "segment_count":0 ,
            },
            )
            s ["doc_score"]+=float (ann .get ("doc_score",0.0 ))
            s ["segment_count"]+=int (ann .get ("segment_count",0 ))


        for ann in annotations [:int (top_n )]:
            leaf_id =ann .get ("competency_id")
            if not isinstance (leaf_id ,str ):
                continue 
            if _level (leaf_id )!=3 :
                continue 
            parent_id =parent_map .get (leaf_id )
            grand_id =parent_map .get (parent_id )if parent_id else None 
            if parent_id and _level (parent_id )==2 :
                _accumulate (level2_stats ,parent_id ,ann )
            if grand_id and _level (grand_id )==1 :
                _accumulate (level1_stats ,grand_id ,ann )

        def _build_agg_rows (stats :Dict [str ,Dict [str ,Any ]])->List [Dict [str ,Any ]]:
            rows :List [Dict [str ,Any ]]=[]
            for i ,(_ ,s )in enumerate (
            sorted (stats .items (),key =lambda kv :kv [1 ]["doc_score"],reverse =True ),
            start =1 ,
            ):
                cid =s ["competency_id"]
                name =full_label_map .get (cid ,cid )
                clean =name .replace ("Раздел:","").replace ("Область:","")
                parts =[p .strip (" .")for p in clean .split (".")if p .strip (" .")]
                display ="\n".join (parts )if len (parts )>1 else clean .strip ()
                rows .append (
                {
                "rank":i ,
                "competency":display ,
                "score":s ["doc_score"],
                "matches":s ["segment_count"],
                }
                )
            return rows 

        level1_rows =_build_agg_rows (level1_stats )
        level2_rows =_build_agg_rows (level2_stats )

        if level1_rows :
            st .subheader ("Топ разделов (1-й уровень)")
            st .dataframe (
            level1_rows ,
            use_container_width =True ,
            hide_index =True ,
            column_config ={
            "rank":st .column_config .NumberColumn ("№",width ="small"),
            "competency":st .column_config .TextColumn ("Раздел"),
            "score":st .column_config .NumberColumn ("Score",format ="%.4f",width ="small"),
            "matches":st .column_config .NumberColumn ("Совпадений (сегментов)",width ="small"),
            },
            )

        if level2_rows :
            st .subheader ("Топ областей (2-й уровень)")
            st .dataframe (
            level2_rows ,
            use_container_width =True ,
            hide_index =True ,
            column_config ={
            "rank":st .column_config .NumberColumn ("№",width ="small"),
            "competency":st .column_config .TextColumn ("Область"),
            "score":st .column_config .NumberColumn ("Score",format ="%.4f",width ="small"),
            "matches":st .column_config .NumberColumn ("Совпадений (сегментов)",width ="small"),
            },
            )

        st .divider ()
        st .subheader ("Детали по сегментам")

        for ann in annotations [:int (top_n )]:
            cid =ann .get ("competency_id")
            if isinstance (cid ,str )and cid in full_label_map :
                full_label =full_label_map [cid ]
            else :
                full_label =ann .get ("competency_full_label",ann .get ("competency_label",""))

            full_label_clean =full_label .replace ("Раздел:","").replace ("Область:","")
            parts =[p .strip (" .")for p in full_label_clean .split (".")if p .strip (" .")]
            if len (parts )>1 :
                display_full_label ="\n".join (parts )
            else :
                display_full_label =full_label_clean .strip ()
            score =float (ann .get ("doc_score",ann .get ("sum_score",ann .get ("max_confidence",0.0 ))))
            matches =ann .get ("matches",[])or []

            with st .expander (f"{display_full_label} — score={score:.3f} — matches={len(matches)}"):
                for m in matches [:int (top_k )]:
                    seg =(m .get ("segment")or "").replace ("\n"," ").strip ()
                    seg_score =float (m .get ("score",0.0 ))
                    st .write (f"- **{seg_score:.3f}**: {seg}")


if __name__ =="__main__":
    main ()

