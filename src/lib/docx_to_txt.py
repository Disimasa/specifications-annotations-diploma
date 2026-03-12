from __future__ import annotations 

import argparse 
import os 
import re 
import tempfile 
from pathlib import Path 
from typing import List 

try :
    import aspose .words as aw 
    HAS_ASPOSE =True 
except ImportError :
    HAS_ASPOSE =False 


_license_loaded =False 


def _load_license ()->None :
    global _license_loaded 
    if _license_loaded or not HAS_ASPOSE :
        return 

    try :

        license_path =Path (__file__ ).resolve ().parents [2 ]/"keys"/"Aspose.WordsforPythonvia.NET.lic"

        if license_path .exists ():
            license =aw .License ()
            license .set_license (str (license_path ))
            _license_loaded =True 

    except Exception :

        pass 


def _remove_watermarks (document :aw .Document )->None :
    try :

        for section in document .sections :

            for header_footer in section .headers_footers :

                nodes =header_footer .get_child_nodes (aw .NodeType .ANY ,True )


                nodes_to_remove =[]
                for node in nodes :

                    if node .node_type ==aw .NodeType .SHAPE :
                        nodes_to_remove .append (node )

                    elif node .node_type ==aw .NodeType .GROUP_SHAPE :
                        nodes_to_remove .append (node )

                    elif node .node_type ==aw .NodeType .PARAGRAPH :
                        para =node 

                        has_only_shapes =True 
                        for child in para .get_child_nodes (aw .NodeType .ANY ,False ):
                            if child .node_type not in (aw .NodeType .SHAPE ,aw .NodeType .GROUP_SHAPE ,aw .NodeType .RUN ):
                                has_only_shapes =False 
                                break 
                        if has_only_shapes :
                            nodes_to_remove .append (node )


                for node in nodes_to_remove :
                    try :
                        node .remove ()
                    except Exception :

                        pass 
    except Exception :

        pass 


def _remove_aspose_watermarks (text :str )->str :
    watermark_patterns =[
    r'Created with an evaluation copy of Aspose\.Words\..*?license/',
    r'Evaluation Only\. Created with Aspose\.Words\..*?Aspose Pty Ltd\.',
    ]

    lines =text .split ('\n')
    cleaned_lines =[]

    for i ,line in enumerate (lines ):

        is_watermark =False 
        for pattern in watermark_patterns :
            if re .search (pattern ,line ,re .IGNORECASE |re .DOTALL ):
                is_watermark =True 
                break 



        if i <3 and re .match (r'^[\s]*\d+[\s]*$',line ):
            is_watermark =True 

        if not is_watermark :
            cleaned_lines .append (line )


    while cleaned_lines and not cleaned_lines [0 ].strip ():
        cleaned_lines .pop (0 )
    while cleaned_lines and not cleaned_lines [-1 ].strip ():
        cleaned_lines .pop ()


    if cleaned_lines and cleaned_lines [0 ]:
        cleaned_lines [0 ]=cleaned_lines [0 ].lstrip ('\ufeff\u200b\u200c\u200d\ufeff')

        if re .match (r'^[\s]*\d+[\s]*$',cleaned_lines [0 ]):
            cleaned_lines .pop (0 )

    return '\n'.join (cleaned_lines )


def _normalize_whitespace (text :str )->str :
    non_breaking_spaces ='\u00A0\u2007\u202F\u2060\uFEFF'

    for nbsp in non_breaking_spaces :
        text =text .replace (nbsp ,' ')

    return text 


def _normalize_bullet_markers (text :str )->str :
    lines =text .split ('\n')
    normalized_lines =[]

    for line in lines :


        normalized_line =re .sub (r'([•◦▪▫○●◉◯◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡·\u00B7o\-])\s{2,}',r'\1 ',line )
        normalized_lines .append (normalized_line )

    return '\n'.join (normalized_lines )


def extract_text (docx_path :Path )->str :
    if not HAS_ASPOSE :
        raise ImportError ("aspose.words не установлен. Установите: pip install aspose-words")


    _load_license ()


    doc =aw .Document (str (docx_path ))


    _remove_watermarks (doc )


    with tempfile .NamedTemporaryFile (mode ='w',suffix ='.txt',delete =False ,encoding ='utf-8')as tmp_file :
        tmp_path =tmp_file .name 

    try :

        doc .save (tmp_path ,aw .SaveFormat .TEXT )


        with open (tmp_path ,'r',encoding ='utf-8')as f :
            text =f .read ()


        text =_remove_aspose_watermarks (text )


        text =_normalize_whitespace (text )


        text =_normalize_bullet_markers (text )

        return text 
    finally :

        if os .path .exists (tmp_path ):
            os .unlink (tmp_path )


def convert_docs_to_txt (source_dir :Path ,target_dir :Path ,encoding :str ="utf-8")->List [Path ]:
    target_dir .mkdir (parents =True ,exist_ok =True )
    written :List [Path ]=[]

    for doc_path in sorted (source_dir .glob ("*.docx")):
        try :
            text =extract_text (doc_path )
            out_path =target_dir /f"{doc_path.stem}.txt"
            out_path .write_text (text ,encoding =encoding )
            written .append (out_path )
        except Exception as e :
            print (f"Ошибка при конвертации {doc_path}: {e}")

    return written 


def main ()->None :
    parser =argparse .ArgumentParser (description ="Конвертация DOCX в TXT используя aspose.words.")
    parser .add_argument (
    "--src",
    type =Path ,
    default =Path ("data/specifications/docs"),
    help ="Каталог с DOCX-файлами",
    )
    parser .add_argument (
    "--dst",
    type =Path ,
    default =Path ("data/specifications/texts"),
    help ="Каталог назначения для TXT",
    )
    parser .add_argument ("--encoding",default ="utf-8",help ="Кодировка выходных файлов")
    args =parser .parse_args ()

    written =convert_docs_to_txt (args .src ,args .dst ,encoding =args .encoding )
    for path in written :
        print (f"Saved {path}")


if __name__ =="__main__":
    main ()
