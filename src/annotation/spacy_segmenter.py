from __future__ import annotations 

import re 
from typing import List 

import spacy 


class SpacySegmenter :
    def __init__ (self ,model_name :str ="ru_core_news_sm")->None :
        try :
            self .nlp =spacy .load (model_name )
        except OSError as exc :
            raise RuntimeError (
            "Модель spaCy не найдена. Установите её командой "
            "`python -m spacy download ru_core_news_sm`."
            )from exc 

    def segment (self ,text :str )->List [str ]:
        blocks =re .split (r"\n\s*\n",text )
        segments :List [str ]=[]

        for raw_block in blocks :
            block =raw_block .strip ()
            if not block :
                continue 

            if self ._looks_like_table (block ):
                segments .extend (self ._collect_table_segments (block ))
                continue 

            block =self ._restore_hyphenation (block )
            doc =self .nlp (block )
            sentences =[self ._normalize_whitespace (sent .text )for sent in doc .sents if sent .text .strip ()]
            sentences =self ._merge_number_markers (sentences )
            segments .extend (sentences )

        return [seg for seg in segments if seg ]

    def _merge_number_markers (self ,sentences :List [str ])->List [str ]:
        merged :List [str ]=[]
        buffer :str |None =None 

        for sentence in sentences :
            if re .fullmatch (r"\d+[\.\)]?",sentence ):
                buffer =f"{buffer} {sentence}".strip ()if buffer else sentence 
                continue 

            if buffer :
                merged .append (f"{buffer} {sentence}".strip ())
                buffer =None 
            else :
                merged .append (sentence )

        if buffer :
            merged .append (buffer )

        return merged 

    def _collect_table_segments (self ,block :str )->List [str ]:
        lines =[self ._normalize_whitespace (line )for line in block .splitlines ()if line .strip ()]
        segments :List [str ]=[]
        chunk :List [str ]=[]

        for line in lines :
            chunk .append (line )
            if len (" ".join (chunk ))>=300 :
                segments .append (" ".join (chunk ).strip ())
                chunk =[]

        if chunk :
            segments .append (" ".join (chunk ).strip ())
        return segments 

    @staticmethod 
    def _normalize_whitespace (text :str )->str :
        text =text .replace ("\xa0"," ")
        text =re .sub (r"\s+"," ",text )
        return text .strip ()

    @staticmethod 
    def _restore_hyphenation (text :str )->str :
        return re .sub (r"(\S)-\s*\n\s*(\S)",r"\1\2",text )

    def _looks_like_table (self ,block :str )->bool :
        lines =[line .strip ()for line in block .splitlines ()if line .strip ()]
        if len (lines )<2 :
            return False 

        table_like =0 
        for line in lines :
            if self ._is_list_like_line (line )or self ._has_many_separators (line ):
                table_like +=1 
        return table_like /len (lines )>=0.5 

    @staticmethod 
    def _is_list_like_line (line :str )->bool :
        return bool (re .match (r"^[-•\d]+[.)]?\s+",line ))

    @staticmethod 
    def _has_many_separators (line :str )->bool :
        separators =line .count (";")+line .count ("|")+line .count ("\t")
        return separators >=2 

