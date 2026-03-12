from __future__ import annotations 

import re 
from dataclasses import dataclass 
from typing import Dict ,List ,Tuple 

from razdel import sentenize 


@dataclass 
class SegmenterConfig :
    max_segment_length :int =400 


class TextSegmenter :
    def __init__ (self ,config :SegmenterConfig |None =None )->None :
        self .config =config or SegmenterConfig ()

    def segment (self ,text :str )->List [str ]:
        blocks =re .split (r"\n\s*\n",text )
        segments :List [str ]=[]

        for raw_block in blocks :
            block =raw_block .strip ()
            if not block :
                continue 



            is_numbered =self ._is_numbered_block (block )
            is_bulleted =self ._is_bulleted_block (block )

            if is_numbered and is_bulleted :

                segments .extend (self ._segment_mixed_list_block (block ))
            elif is_numbered :
                segments .extend (self ._segment_numbered_block (block ))
            elif is_bulleted :
                segments .extend (self ._segment_bulleted_block (block ))
            elif self ._looks_like_table (block ):
                segments .extend (self ._collect_table_segments (block ))
            else :
                segments .extend (self ._split_sentences (block ))

        cleaned =[self ._normalize_whitespace (seg )for seg in segments if seg .strip ()]


        cleaned =[self ._remove_leading_bullets (seg )for seg in cleaned if seg .strip ()]
        return cleaned 

    def _collect_table_segments (self ,block :str )->List [str ]:
        lines =[self ._normalize_whitespace (line )for line in block .splitlines ()if line .strip ()]
        segments :List [str ]=[]
        chunk :List [str ]=[]

        for line in lines :
            chunk .append (line )
            if len (" ".join (chunk ))>=self .config .max_segment_length :
                segments .append (" ".join (chunk ).strip ())
                chunk =[]

        if chunk :
            segments .append (" ".join (chunk ).strip ())
        return segments 

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

        return bool (re .match (r"^[-•\do·\u00B7]+[.)]?\s+",line ))

    @staticmethod 
    def _has_many_separators (line :str )->bool :
        separators =line .count (";")+line .count ("|")+line .count ("\t")
        return separators >=2 

    @staticmethod 
    def _normalize_whitespace (text :str )->str :
        return re .sub (r"\s+"," ",text ).strip ()

    @staticmethod 
    def _remove_leading_bullets (text :str )->str :
        text =re .sub (r'^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+','',text )

        text =re .sub (r'^\s+[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+','',text )


        if text .startswith ('o '):
            text =text [2 :]

        text =re .sub (r'^(\s+)o\s+',r'\1',text )

        if text .startswith ('· ')or text .startswith ('\u00B7 '):
            text =text [2 :]

        text =re .sub (r'^(\s+)[·\u00B7]\s+',r'\1',text )
        return text .strip ()

    def _split_sentences (self ,text :str )->List [str ]:

        sentences =[s .text .strip ()for s in sentenize (text )if s .text .strip ()]
        if not sentences :
            return [self ._normalize_whitespace (text )]
        sentences =self ._merge_quoted_fragments (sentences )
        return [self ._normalize_whitespace (sentence )for sentence in sentences ]

    def _merge_quoted_fragments (self ,sentences :List [str ])->List [str ]:
        quote_balance :Dict [Tuple [str ,str ],int ]={
        ("«","»"):0 ,
        ("„","“"):0 ,
        ("“","”"):0 ,
        }
        symmetric_quotes ={'"':False ,"'":False }
        merged :List [str ]=[]
        buffer :List [str ]=[]

        for sentence in sentences :
            buffer .append (sentence )
            self ._update_quote_state (sentence ,quote_balance ,symmetric_quotes )
            if not self ._inside_quotes (quote_balance ,symmetric_quotes ):
                merged .append (" ".join (buffer ).strip ())
                buffer =[]

        if buffer :
            merged .append (" ".join (buffer ).strip ())
        return merged 

    @staticmethod 
    def _update_quote_state (
    sentence :str ,
    quote_balance :Dict [Tuple [str ,str ],int ],
    symmetric_quotes :Dict [str ,bool ],
    )->None :
        for pair ,balance in quote_balance .items ():
            open_char ,close_char =pair 
            quote_balance [pair ]=balance +sentence .count (open_char )-sentence .count (close_char )
        for quote_char ,is_open in symmetric_quotes .items ():
            if sentence .count (quote_char )%2 ==1 :
                symmetric_quotes [quote_char ]=not is_open 

    @staticmethod 
    def _inside_quotes (
    quote_balance :Dict [Tuple [str ,str ],int ],
    symmetric_quotes :Dict [str ,bool ],
    )->bool :
        if any (balance >0 for balance in quote_balance .values ()):
            return True 
        if any (symmetric_quotes .values ()):
            return True 
        return False 

    def _is_numbered_block (self ,block :str )->bool :
        matches =re .findall (r"(?:^|\s)\d+[\.\)]\s",block )
        return len (matches )>=2 

    def _is_bulleted_block (self ,block :str )->bool :
        lines =block .splitlines ()
        if len (lines )<2 :
            return False 



        bullet_pattern =re .compile (r"^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+")
        bullet_count =sum (1 for line in lines if bullet_pattern .match (line .strip ()))


        return bullet_count >=2 

    def _segment_numbered_block (self ,block :str )->List [str ]:







        pattern =re .compile (r"(?<!\S)(\d+(?:\.\d+)*[\.\)])\s+|(?<!\S)(\d+(?:\.\d+)+)\s+")
        matches =list (pattern .finditer (block ))
        if not matches :
            return self ._split_sentences (block )

        segments :List [str ]=[]
        first_start =matches [0 ].start ()
        if first_start >0 :
            prefix =block [:first_start ].strip ()
            if prefix :
                segments .extend (self ._split_sentences (prefix ))


        numbered_items :List [Tuple [int ,str ,str ]]=[]
        for match in matches :

            prefix =match .group (1 )or match .group (2 )

            number_part =prefix .rstrip (".)")
            numbered_items .append ((match .start (),prefix ,number_part ))



        split_positions =[numbered_items [0 ][0 ]]

        for i in range (1 ,len (numbered_items )):
            prev_number =numbered_items [i -1 ][2 ]
            curr_number =numbered_items [i ][2 ]




            is_subitem =curr_number .startswith (prev_number +".")




            split_positions .append (numbered_items [i ][0 ])

        split_positions .append (len (block ))



        for idx ,start in enumerate (split_positions [:-1 ]):
            end =split_positions [idx +1 ]
            chunk =block [start :end ].strip ()
            if not chunk :
                continue 



            pattern =re .compile (r"(?<!\S)(\d+(?:\.\d+)*[\.\)])\s+|(?<!\S)(\d+(?:\.\d+)+)\s+")
            chunk_matches =list (pattern .finditer (chunk ))

            if len (chunk_matches )>1 :


                for chunk_idx ,match in enumerate (chunk_matches ):
                    chunk_start =match .start ()

                    if chunk_idx +1 <len (chunk_matches ):
                        chunk_end =chunk_matches [chunk_idx +1 ].start ()
                    else :
                        chunk_end =len (chunk )

                    sub_chunk =chunk [chunk_start :chunk_end ].strip ()
                    if sub_chunk :
                        sub_chunk =self ._normalize_whitespace (sub_chunk )
                        segments .append (sub_chunk )
            else :



                lines_in_chunk =chunk .splitlines ()
                if len (lines_in_chunk )>1 :

                    first_line =lines_in_chunk [0 ].strip ()
                    if first_line :

                        second_line =lines_in_chunk [1 ].strip ()
                        is_continuation =(
                        second_line and 
                        not re .match (r"^\d+[\.\)]\s+",second_line )and 
                        not re .match (r"^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+",second_line )
                        )

                        if is_continuation :


                            combined =' '.join (line .strip ()for line in lines_in_chunk if line .strip ())
                            combined =self ._normalize_whitespace (combined )
                            segments .append (combined )
                        else :

                            segments .extend (self ._process_chunk (chunk ))
                else :

                    segments .extend (self ._process_chunk (chunk ))

        return segments 

    def _segment_bulleted_block (self ,block :str )->List [str ]:
        segments :List [str ]=[]
        lines =block .splitlines ()


        all_positions =[]


        current_pos =0 
        for line in lines :
            line_stripped =line .strip ()


            match =re .match (r"^([•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7])\s+",line_stripped )
            if match :
                all_positions .append (current_pos )
            current_pos +=len (line )+1 



        pattern_after_separator =re .compile (r"([,;:])\s+([•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7])\s+")
        for match in pattern_after_separator .finditer (block ):

            marker_pos =match .end ()-len (match .group (2 ))-1 
            all_positions .append (marker_pos )

        if not all_positions :
            return self ._split_sentences (block )

        all_positions .sort ()
        first_start =all_positions [0 ]


        if first_start >0 :
            prefix =block [:first_start ].strip ()
            if prefix :
                segments .extend (self ._split_sentences (prefix ))


        all_positions .append (len (block ))

        for idx ,start in enumerate (all_positions [:-1 ]):
            end =all_positions [idx +1 ]
            chunk =block [start :end ].strip ()
            if not chunk :
                continue 


            chunk =re .sub (r"^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+","",chunk )

            chunk =re .sub (r"^[\s,;:]+","",chunk )
            if not chunk :
                continue 



            lines_in_chunk =chunk .splitlines ()
            if len (lines_in_chunk )>1 :

                first_line =lines_in_chunk [0 ].strip ()
                if first_line :
                    segments .extend (self ._process_chunk (first_line ))


                remaining_text ='\n'.join (lines_in_chunk [1 :]).strip ()
                if remaining_text :

                    first_remaining_line =lines_in_chunk [1 ].strip ()
                    is_list_continuation =(
                    re .match (r"^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+",first_remaining_line )or 
                    re .match (r"^\d+[\.\)]\s+",first_remaining_line )
                    )

                    if not is_list_continuation :

                        segments .extend (self ._split_sentences (remaining_text ))
                    else :

                        segments .extend (self ._process_chunk (chunk ))
            else :

                segments .extend (self ._process_chunk (chunk ))

        return segments 

    def _segment_mixed_list_block (self ,block :str )->List [str ]:
        segments :List [str ]=[]


        numbered_pattern =re .compile (r"(?<!\S)(\d+(?:\.\d+)*[\.\)])\s+|(?<!\S)(\d+(?:\.\d+)+)\s+")
        bulleted_pattern =re .compile (r"^([•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7])\s+",re .MULTILINE )
        bulleted_pattern_separator =re .compile (r"([,;:])\s+([•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7])\s+")

        numbered_matches =list (numbered_pattern .finditer (block ))
        bulleted_matches_start =list (bulleted_pattern .finditer (block ))
        bulleted_matches_sep =list (bulleted_pattern_separator .finditer (block ))


        all_positions =[]
        for match in numbered_matches :
            all_positions .append ((match .start (),'numbered'))
        for match in bulleted_matches_start :
            all_positions .append ((match .start (),'bulleted'))
        for match in bulleted_matches_sep :
            marker_pos =match .end ()-len (match .group (2 ))-1 
            all_positions .append ((marker_pos ,'bulleted'))

        if not all_positions :
            return self ._split_sentences (block )

        all_positions .sort (key =lambda x :x [0 ])
        first_start =all_positions [0 ][0 ]


        if first_start >0 :
            prefix =block [:first_start ].strip ()
            if prefix :
                segments .extend (self ._split_sentences (prefix ))


        positions =[pos for pos ,_ in all_positions ]
        positions .append (len (block ))

        for idx ,start in enumerate (positions [:-1 ]):
            end =positions [idx +1 ]
            chunk =block [start :end ].strip ()
            if not chunk :
                continue 


            chunk =re .sub (r"^[•\-\u2022\u2023\u25E6\u25AA\u25CFo·\u00B7]\s+","",chunk )
            chunk =re .sub (r"^[\s,;:]+","",chunk )
            if chunk :
                segments .extend (self ._process_chunk (chunk ))

        return segments 

    def _process_chunk (self ,chunk :str )->List [str ]:


        pattern =re .compile (r"(?<!\S)(\d+(?:\.\d+)*[\.\)])\s+|(?<!\S)(\d+(?:\.\d+)+)\s+")
        matches =list (pattern .finditer (chunk ))


        if len (matches )>1 :
            segments :List [str ]=[]
            positions =[match .start ()for match in matches ]
            positions .append (len (chunk ))

            for idx ,start in enumerate (positions [:-1 ]):
                end =positions [idx +1 ]
                sub_chunk =chunk [start :end ].strip ()
                if not sub_chunk :
                    continue 


                sub_chunk =self ._normalize_whitespace (sub_chunk )


                if self ._looks_like_table (sub_chunk ):
                    prefix ,table_part =self ._split_prefix_table (sub_chunk )
                    if prefix .strip ():

                        segments .append (prefix .strip ())
                    if table_part .strip ():
                        segments .extend (self ._collect_table_segments (table_part ))
                else :


                    segments .append (sub_chunk )
            return segments 


        if self ._looks_like_table (chunk ):
            prefix ,table_part =self ._split_prefix_table (chunk )
            result :List [str ]=[]
            if prefix .strip ():
                result .extend (self ._split_sentences (prefix ))
            table_text =table_part if table_part .strip ()else chunk 
            result .extend (self ._collect_table_segments (table_text ))
            return result 
        return self ._split_sentences (chunk )

    def _split_prefix_table (self ,text :str )->Tuple [str ,str ]:
        lines =text .splitlines ()
        table_start =None 
        for idx ,line in enumerate (lines ):
            if self ._is_table_line (line ):
                table_start =idx 
                break 
        if table_start is None :
            return text ,""
        prefix ="\n".join (lines [:table_start ])
        table_part ="\n".join (lines [table_start :])
        return prefix ,table_part 

    def _is_table_line (self ,line :str )->bool :
        stripped =line .strip ()
        if not stripped :
            return False 
        return ("|"in stripped )or self ._has_many_separators (stripped )or bool (re .search (r"\s{2,}\S+\s{2,}",stripped ))

