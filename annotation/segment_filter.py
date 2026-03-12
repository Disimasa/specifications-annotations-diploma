import re 
from typing import List 


class SegmentFilter :

    NON_INFORMATIVE_PHRASES ={
    "проект",
    "утверждаю",
    "техническое задание",
    "согласовано",
    "приложение",
    "содержание",
    "раздел",
    "задачи:",
    "задача:",
    "цель:",
    "цель",
    }


    SECTION_HEADER_PATTERNS =[
    r"^РАЗДЕЛ\s+\d+\.?\s*$",
    r"^Раздел\s+\d+\.?\s*$",
    r"^РАЗДЕЛ\s+\d+\.?\s+[А-ЯЁ\s]+$",
    ]


    DATE_PATTERNS =[
    r"^\d{1,2}\.\d{1,2}\.\d{2,4}$",
    r"^\d{1,2}\.\d{1,2}\.\d{2,4}\s*$",
    ]


    LOCATION_DATE_PATTERNS =[
    r"^[А-ЯЁ][а-яё\s]+,\s*г\.\s*[А-ЯЁ][а-яё]+\s+\d{4}$",
    r"^[А-ЯЁ][а-яё]+\s+\d{4}$",
    ]


    MIN_INFORMATIVE_LENGTH =15 

    def __init__ (
    self ,
    min_length :int =MIN_INFORMATIVE_LENGTH ,
    filter_headers :bool =True ,
    filter_dates :bool =True ,
    filter_phrases :bool =True 
    ):
        self .min_length =min_length 
        self .filter_headers =filter_headers 
        self .filter_dates =filter_dates 
        self .filter_phrases =filter_phrases 

    def is_non_informative (self ,segment :str )->bool :
        segment =segment .strip ()


        if not segment :
            return True 


        if len (segment )<self .min_length :
            return True 


        if self .filter_phrases :
            segment_lower =segment .lower ().strip ()

            if segment_lower in self .NON_INFORMATIVE_PHRASES :
                return True 


            if segment_lower .rstrip (":")in self .NON_INFORMATIVE_PHRASES :
                return True 


        if self .filter_headers :
            for pattern in self .SECTION_HEADER_PATTERNS :
                if re .match (pattern ,segment ,re .IGNORECASE ):
                    return True 



            section_match =re .match (r"^РАЗДЕЛ\s+\d+\.?\s+([А-ЯЁ\s]+)$",segment ,re .IGNORECASE )
            if section_match :

                title_part =section_match .group (1 ).strip ()
                if len (title_part )<50 and title_part .isupper ():

                    if not any (char in segment for char in [".",",",":",";"]):
                        return True 


        if self .filter_dates :
            for pattern in self .DATE_PATTERNS :
                if re .match (pattern ,segment ):
                    return True 

            for pattern in self .LOCATION_DATE_PATTERNS :
                if re .match (pattern ,segment ):
                    return True 


        if re .match (r"^[\d\s\.\,\-\:]+$",segment ):
            return True 




        return False 

    def filter_segments (self ,segments :List [str ])->List [str ]:
        filtered =[]
        for segment in segments :
            if not self .is_non_informative (segment ):
                filtered .append (segment )
        return filtered 

