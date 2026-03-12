from pathlib import Path 
from annotation import annotate_document 

base_dir =Path (__file__ ).parent 
ontology_path =base_dir /"data"/"ontology_filtered.json"
text_path =base_dir /"data"/"specifications"/"texts"/"2024.05.15_Проект ТЗ_аргон.txt"

print ("="*80 )
print ("Тест БЕЗ контекстного окна (max_segment_length_for_context=0):")
print ("="*80 )
annotations_no_context =annotate_document (
text_path ,
ontology_path ,
threshold =0.45 ,
top_k =3 ,
max_segment_length_for_context =0 
)


for ann in annotations_no_context [:3 ]:
    print (f"\n{ann['competency_label']} (max {ann['max_confidence']:.2f}):")
    for match in ann ["matches"][:2 ]:
        snippet =match ["segment"].replace ("\n"," ")[:100 ]
        print (f"  - {match['score']:.2f}: {snippet}...")

print ("\n"+"="*80 )
print ("Тест С контекстным окном (max_segment_length_for_context=100):")
print ("="*80 )
annotations_with_context =annotate_document (
text_path ,
ontology_path ,
threshold =0.45 ,
top_k =3 ,
max_segment_length_for_context =100 
)


for ann in annotations_with_context [:3 ]:
    print (f"\n{ann['competency_label']} (max {ann['max_confidence']:.2f}):")
    for match in ann ["matches"][:2 ]:
        snippet =match ["segment"].replace ("\n"," ")[:100 ]
        print (f"  - {match['score']:.2f}: {snippet}...")

