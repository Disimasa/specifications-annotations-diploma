from pathlib import Path 
from lib .docx_to_txt import convert_docs_to_txt 


def main ():
    base_dir =Path (__file__ ).resolve ().parents [2 ]
    source_dir =base_dir /"data"/"specifications"/"docs"
    target_dir =base_dir /"data"/"specifications"/"texts"

    if not source_dir .exists ():
        print (f"Ошибка: папка {source_dir} не существует")
        return 

    print (f"Конвертация DOCX файлов из {source_dir} в {target_dir}...")

    written_files =convert_docs_to_txt (source_dir ,target_dir ,encoding ="utf-8")

    if written_files :
        print (f"\nУспешно сконвертировано {len(written_files)} файлов:")
        for file_path in written_files :
            print (f"  - {file_path.name}")
    else :
        print ("\nНе найдено DOCX файлов для конвертации")


if __name__ =="__main__":
    main ()

