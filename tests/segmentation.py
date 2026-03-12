from pathlib import Path 

from annotation .segmenter import TextSegmenter 
from annotation .spacy_segmenter import SpacySegmenter 


def main ()->None :
    base_dir =Path (__file__ ).resolve ().parent .parent 
    text_path =base_dir /"data"/"specifications"/"texts"/"ТЗ_1518025 Определение характеристик мелющих шаров_2023_ТЗ №4.txt"

    text =text_path .read_text (encoding ="utf-8")
    print ("=== Razdel segmenter ===")
    razdel_segments =TextSegmenter ().segment (text )
    for idx ,segment in enumerate (razdel_segments ,1 ):
        print (f"{idx}: {segment}")







if __name__ =="__main__":
    main ()

