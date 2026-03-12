from __future__ import annotations 

import json 
import re 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Tuple 


PROJECT_DIR =Path (__file__ ).resolve ().parents [2 ]
DEFAULT_GOLD_PATH =PROJECT_DIR /"data"/"gold"/"test_set_manual_draft.jsonl"


def _clean_json_line (s :str )->str :
    s =s .strip ()
    if not s :
        return s 

    s =re .sub (r",\s*$","",s )

    s =re .sub (r"\"\,([A-Za-z_][A-Za-z0-9_]*)\"\s*:",r'"\1":',s )

    s =re .sub (r",\s*([}\]])",r"\1",s )
    return s 


def _normalize_obj (obj :Dict [str ,Any ])->Dict [str ,Any ]:

    if isinstance (obj .get ("gold_codes"),list ):
        cleaned :List [str ]=[]
        for x in obj ["gold_codes"]:
            if not isinstance (x ,str ):
                continue 
            c =x .strip ().lstrip (" ,;")
            if c :
                cleaned .append (c )
        obj ["gold_codes"]=cleaned 


    if ",evidence"in obj and "evidence"not in obj :
        obj ["evidence"]=obj .pop (",evidence")

    return obj 


def fix_gold_jsonl (path :Path )->Tuple [int ,int ]:
    ok :List [dict ]=[]
    dropped =0 

    for raw_line in path .read_text (encoding ="utf-8").splitlines ():
        line =raw_line .strip ()
        if not line :
            continue 

        cleaned =_clean_json_line (line )
        try :
            obj =json .loads (cleaned )
        except json .JSONDecodeError :
            dropped +=1 
            continue 

        if not isinstance (obj ,dict ):
            dropped +=1 
            continue 

        obj =_normalize_obj (obj )
        ok .append (obj )


    path .write_text (
    "\n".join (json .dumps (o ,ensure_ascii =False )for o in ok )+("\n"if ok else ""),
    encoding ="utf-8",
    )
    return (len (ok ),dropped )


def main ()->None :
    path =DEFAULT_GOLD_PATH 
    if not path .exists ():
        raise FileNotFoundError (f"GOLD файл не найден: {path}")

    ok ,dropped =fix_gold_jsonl (path )
    print (f"OK: переписано строк: {ok}, пропущено (битых): {dropped}")
    print (f"Файл: {path}")


if __name__ =="__main__":
    main ()

