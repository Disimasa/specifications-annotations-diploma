import json 
from pathlib import Path 

try :
    import pytest 
    HAS_PYTEST =True 
except ImportError :
    HAS_PYTEST =False 

from annotation .segmenter import TextSegmenter 


def get_test_cases ():
    base_dir =Path (__file__ ).parent .parent 
    texts_dir =base_dir /"data"/"specifications"/"texts"
    reference_dir =base_dir /"data"/"segmentations"/"reference"

    test_cases =[]


    for txt_file in texts_dir .glob ("*.txt"):

        json_file =reference_dir /(txt_file .stem +".json")
        if json_file .exists ():
            test_cases .append ((txt_file ,json_file ))

    return test_cases 


if HAS_PYTEST :
    @pytest .mark .parametrize ("txt_file,reference_json_file",get_test_cases ())
    def test_segmentation_matches_reference (txt_file :Path ,reference_json_file :Path ):
        with reference_json_file .open ("r",encoding ="utf-8")as f :
            reference_segments =json .load (f )


        text =txt_file .read_text (encoding ="utf-8")
        segmenter =TextSegmenter ()
        actual_segments =segmenter .segment (text )


        base_dir =Path (__file__ ).parent .parent 
        test_res_dir =base_dir /"data"/"segmentations"/"test_res"
        test_res_dir .mkdir (parents =True ,exist_ok =True )
        result_file =test_res_dir /(txt_file .stem +".json")
        with result_file .open ("w",encoding ="utf-8")as f :
            json .dump (actual_segments ,f ,ensure_ascii =False ,indent =2 )


        assert len (actual_segments )==len (reference_segments ),(
        f"Количество сегментов не совпадает для {txt_file.name}:\n"
        f"  Ожидалось: {len(reference_segments)}\n"
        f"  Получено: {len(actual_segments)}"
        )


        mismatches =[]
        for i ,(actual ,reference )in enumerate (zip (actual_segments ,reference_segments )):
            if actual !=reference :
                mismatches .append ({
                "index":i ,
                "expected":reference ,
                "actual":actual ,
                })

        if mismatches :
            error_msg =f"Найдены несоответствия в сегментации для {txt_file.name}:\n\n"
            for mismatch in mismatches [:10 ]:
                error_msg +=f"Сегмент {mismatch['index']}:\n"
                error_msg +=f"  Ожидалось: {repr(mismatch['expected'])}\n"
                error_msg +=f"  Получено:   {repr(mismatch['actual'])}\n\n"
            if len (mismatches )>10 :
                error_msg +=f"... и еще {len(mismatches) - 10} несоответствий\n"
            pytest .fail (error_msg )


if __name__ =="__main__":

    test_cases =get_test_cases ()
    if not test_cases :
        print ("Не найдено тестовых случаев (txt файлы с соответствующими эталонными JSON)")
        exit (1 )

    print (f"Найдено {len(test_cases)} тестовых случаев\n")


    base_dir =Path (__file__ ).parent .parent 
    test_res_dir =base_dir /"data"/"segmentations"/"test_res"
    test_res_dir .mkdir (parents =True ,exist_ok =True )

    segmenter =TextSegmenter ()
    all_passed =True 

    for txt_file ,reference_json_file in test_cases :
        print (f"Тестирование: {txt_file.name}...")

        try :

            with reference_json_file .open ("r",encoding ="utf-8")as f :
                reference_segments =json .load (f )


            text =txt_file .read_text (encoding ="utf-8")
            actual_segments =segmenter .segment (text )


            result_file =test_res_dir /(txt_file .stem +".json")
            with result_file .open ("w",encoding ="utf-8")as f :
                json .dump (actual_segments ,f ,ensure_ascii =False ,indent =2 )


            if len (actual_segments )!=len (reference_segments ):
                print (f"  ✗ Количество сегментов не совпадает: ожидалось {len(reference_segments)}, получено {len(actual_segments)}")
                all_passed =False 
                continue 

            mismatches =[]
            for i ,(actual ,reference )in enumerate (zip (actual_segments ,reference_segments )):
                if actual !=reference :
                    mismatches .append ((i ,reference ,actual ))

            if mismatches :
                print (f"  ✗ Найдено {len(mismatches)} несоответствий:")
                for idx ,expected ,actual in mismatches [:5 ]:
                    print (f"    Сегмент {idx}:")
                    print (f"      Ожидалось: {repr(expected[:100])}...")
                    print (f"      Получено:   {repr(actual[:100])}...")
                if len (mismatches )>5 :
                    print (f"    ... и еще {len(mismatches) - 5} несоответствий")
                all_passed =False 
            else :
                print (f"  ✓ Сегментация совпадает ({len(actual_segments)} сегментов)")

        except Exception as e :
            print (f"  ✗ Ошибка: {e}")
            all_passed =False 

    print (f"\nРезультаты сегментации сохранены в: {test_res_dir}")

    if all_passed :
        print (f"\n✓ Все тесты пройдены!")
        exit (0 )
    else :
        print (f"\n✗ Некоторые тесты не пройдены")
        exit (1 )

