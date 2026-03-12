from pathlib import Path 
import json 

from annotation .segmenter import TextSegmenter 


def main ()->None :
    base_dir =Path (__file__ ).resolve ().parents [2 ]
    texts_dir =base_dir /"data"/"specifications"/"texts"
    jsons_dir =base_dir /"data"/"segmentations"/"reference"


    jsons_dir .mkdir (parents =True ,exist_ok =True )


    txt_files =list (texts_dir .glob ("*.txt"))

    if not txt_files :
        print (f"Не найдено .txt файлов в {texts_dir}")
        return 

    print (f"Найдено {len(txt_files)} файлов для обработки")
    print (f"Выходная папка: {jsons_dir}\n")

    segmenter =TextSegmenter ()

    for txt_file in txt_files :
        print (f"Обработка: {txt_file.name}...")

        try :

            text =txt_file .read_text (encoding ="utf-8")


            segments =segmenter .segment (text )


            json_filename =txt_file .stem +".json"
            json_path =jsons_dir /json_filename 

            with json_path .open ("w",encoding ="utf-8")as f :
                json .dump (segments ,f ,ensure_ascii =False ,indent =2 )

            print (f"  ✓ Сохранено: {json_path.name} ({len(segments)} сегментов)")

        except Exception as e :
            print (f"  ✗ Ошибка при обработке {txt_file.name}: {e}")
            continue 

    print (f"\nГотово! Обработано {len(txt_files)} файлов.")


if __name__ =="__main__":
    main ()

