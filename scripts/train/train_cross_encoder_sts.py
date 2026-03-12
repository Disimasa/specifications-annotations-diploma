import json 
import time 
from pathlib import Path 
from typing import List 

import torch 
from datasets import load_dataset 
from sentence_transformers import CrossEncoder ,InputExample 
from sentence_transformers .cross_encoder .evaluation import (
CEBinaryClassificationEvaluator ,
CECorrelationEvaluator ,
)
from torch .utils .data import DataLoader 


try :
    from train_cross_encoder import load_rusbeir_dataset 
except ImportError :

    def load_rusbeir_dataset (hf_repo :str ,hf_repo_qrels :str ,split :str ="train"):
        from datasets import load_dataset 

        print (f"Загрузка датасета {hf_repo} (split={split})...")

        try :
            corpus_ds =load_dataset (hf_repo ,"corpus",split =split if split !="test"else "test")
            queries_ds =load_dataset (hf_repo ,"queries",split =split if split !="test"else "test")
            qrels_ds =load_dataset (hf_repo_qrels ,split =split if split !="test"else "test")
        except Exception as e :
            raise Exception (f"Не удалось загрузить датасет: {e}")

        corpus ={}
        queries ={}
        qrels ={}

        for item in corpus_ds :
            doc_id =str (item .get ("_id",item .get ("id","")))
            if not doc_id :
                continue 
            text =item .get ("text","")
            if not text :
                title =item .get ("title","")
                body =item .get ("body",item .get ("processed_text",""))
                text =f"{title} {body}".strip ()
            if text :
                corpus [doc_id ]={"text":text }

        for item in queries_ds :
            query_id =str (item .get ("_id",item .get ("id","")))
            if not query_id :
                continue 
            query_text =item .get ("text",item .get ("query",""))
            if query_text :
                queries [query_id ]=query_text 

        for item in qrels_ds :
            query_id =str (item .get ("query-id",item .get ("query_id","")))
            doc_id =str (item .get ("corpus-id",item .get ("doc_id",item .get ("corpus_id",""))))
            if not query_id or not doc_id :
                continue 
            score =item .get ("score",1 )
            if query_id not in qrels :
                qrels [query_id ]={}
            qrels [query_id ][doc_id ]=score 

        return corpus ,queries ,qrels 


CONFIG ={





"base_model":"models/cross-encoder-rusbeir/final_model",


"output_dir":"models/cross-encoder-sts",
"datasets":[
{
"name":"stsb_multi_mt_ru",
"hf_repo":"stsb_multi_mt",
"language":"ru",
"use_train":True ,
"use_dev":True ,
"use_test":True ,
"weight":1.0 ,
},




{
"name":"stsb_multi_mt_slavic",
"hf_repo":"stsb_multi_mt",
"languages":["pl"],
"use_train":True ,
"use_dev":False ,
"use_test":False ,
"weight":0.5 ,
},





















],
"training":{
"num_epochs":3 ,
"batch_size":16 ,


"learning_rate":1e-5 ,
"warmup_steps":500 ,
"max_length":512 ,
"evaluation_steps":5000 ,
},




}


def load_sts_dataset (hf_repo :str ,language :str ="ru",split :str ="train"):
    print (f"Загрузка STS датасета {hf_repo} (language={language}, split={split})...")

    try :

        dataset =load_dataset (hf_repo ,language ,split =split )
    except Exception as e :
        print (f"Ошибка загрузки: {e}")

        try :
            dataset =load_dataset (hf_repo ,split =split )
        except Exception as e2 :
            raise Exception (f"Не удалось загрузить датасет: {e}, {e2}")

    examples =[]

    for item in dataset :

        sentence1 =item .get ("sentence1",item .get ("text1",""))
        sentence2 =item .get ("sentence2",item .get ("text2",""))
        score =item .get ("similarity_score",item .get ("score",0 ))

        if not sentence1 or not sentence2 :
            continue 



        normalized_score =float (score )/5.0 if score >0 else 0.0 

        examples .append (InputExample (texts =[sentence1 ,sentence2 ],label =normalized_score ))

    print (f"Загружено {len(examples)} примеров")
    return examples 


def create_evaluation_data_sts (examples :List [InputExample ],num_samples :int =1000 ):
    import random 


    sample_examples =random .sample (examples ,min (num_samples ,len (examples )))

    sentence_pairs =[]
    labels =[]

    for example in sample_examples :
        sentence_pairs .append (example .texts )
        labels .append (example .label )

    return sentence_pairs ,labels 


def convert_ir_to_sts_format (
corpus :dict ,queries :dict ,qrels :dict ,
num_negatives :int =1 ,
use_relevance_scores :bool =True 
)->List [InputExample ]:
    examples =[]
    all_doc_ids =list (corpus .keys ())
    import random 

    print ("Преобразование IR датасета в STS формат...")

    for query_id ,relevant_docs in qrels .items ():
        if query_id not in queries :
            continue 

        query_text =queries [query_id ]


        for doc_id ,relevance_score in relevant_docs .items ():
            if doc_id not in corpus or relevance_score <=0 :
                continue 

            doc_text =corpus [doc_id ]["text"]


            if use_relevance_scores and relevance_score >1 :

                similarity_score =min (float (relevance_score )/5.0 ,1.0 )
            else :


                similarity_score =0.8 +random .uniform (0 ,0.2 )

            examples .append (InputExample (texts =[query_text ,doc_text ],label =similarity_score ))


        relevant_doc_ids =set (relevant_docs .keys ())
        negative_candidates =[d for d in all_doc_ids if d not in relevant_doc_ids ]

        num_neg =min (num_negatives *len (relevant_docs ),len (negative_candidates ))
        negative_docs =random .sample (negative_candidates ,num_neg )

        for doc_id in negative_docs :
            doc_text =corpus [doc_id ]["text"]


            similarity_score =random .uniform (0.0 ,0.2 )
            examples .append (InputExample (texts =[query_text ,doc_text ],label =similarity_score ))

    positive_count =sum (1 for e in examples if e .label >=0.5 )
    negative_count =len (examples )-positive_count 
    print (f"Создано {len(examples)} примеров в STS формате ({positive_count} с высоким сходством, {negative_count} с низким)")
    return examples 


def main ():
    import argparse 

    parser =argparse .ArgumentParser (description ="Обучение cross-encoder на STS датасетах")
    parser .add_argument (
    "--resume_from",
    type =str ,
    default =None ,
    help ="Путь к checkpoint для продолжения обучения",
    )
    args =parser .parse_args ()

    print ("="*80 )
    print ("Обучение Cross-Encoder для Semantic Textual Similarity (STS)")
    print ("="*80 )
    print (f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch .cuda .is_available ():
        print (f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print ()


    output_dir =Path (CONFIG ["output_dir"])
    output_dir .mkdir (parents =True ,exist_ok =True )
    checkpoint_dir =output_dir /"checkpoints"
    checkpoint_dir .mkdir (exist_ok =True )


    all_train_examples =[]
    test_evaluators =[]

    for dataset_config in CONFIG ["datasets"]:
        dataset_weight =dataset_config .get ("weight",1.0 )


        if dataset_config .get ("type")=="ir_to_sts":
            try :
                corpus ,queries ,qrels =load_rusbeir_dataset (
                dataset_config ["hf_repo"],
                dataset_config ["hf_repo_qrels"],
                split ="train"
                )
                ir_examples =convert_ir_to_sts_format (
                corpus ,
                queries ,
                qrels ,
                num_negatives =dataset_config .get ("num_negatives",1 ),
                use_relevance_scores =True 
                )


                if dataset_weight <1.0 :
                    num_samples =int (len (ir_examples )*dataset_weight )
                    import random 
                    ir_examples =random .sample (ir_examples ,num_samples )

                all_train_examples .extend (ir_examples )
                print (f"Добавлено {len(ir_examples)} примеров из {dataset_config['name']} (IR->STS, weight={dataset_weight})")
            except Exception as e :
                print (f"Не удалось загрузить IR датасет {dataset_config['name']}: {e}")
            continue 


        if "language"in dataset_config :
            if dataset_config .get ("use_train",False ):
                train_examples =load_sts_dataset (
                dataset_config ["hf_repo"],
                dataset_config ["language"],
                split ="train"
                )


                if dataset_weight <1.0 :
                    num_samples =int (len (train_examples )*dataset_weight )
                    import random 
                    train_examples =random .sample (train_examples ,num_samples )

                all_train_examples .extend (train_examples )
                print (f"Добавлено {len(train_examples)} примеров из {dataset_config['name']} (train, weight={dataset_weight})")

            if dataset_config .get ("use_dev",False ):
                dev_examples =load_sts_dataset (
                dataset_config ["hf_repo"],
                dataset_config ["language"],
                split ="dev"
                )
                all_train_examples .extend (dev_examples )
                print (f"Добавлено {len(dev_examples)} примеров из {dataset_config['name']} (dev)")

            if dataset_config .get ("use_test",False ):
                try :
                    test_examples =load_sts_dataset (
                    dataset_config ["hf_repo"],
                    dataset_config ["language"],
                    split ="test"
                    )
                    if test_examples :
                        sentence_pairs ,labels =create_evaluation_data_sts (test_examples )
                        evaluator =CECorrelationEvaluator (
                        sentence_pairs =sentence_pairs ,
                        scores =labels ,
                        name =dataset_config ["name"],
                        )
                        test_evaluators .append (evaluator )
                except Exception as e :
                    print (f"Не удалось загрузить тестовый набор для {dataset_config['name']}: {e}")


        elif "languages"in dataset_config :
            if dataset_config .get ("use_train",False ):
                train_examples =load_sts_dataset_multilang (
                dataset_config ["hf_repo"],
                dataset_config ["languages"],
                split ="train"
                )


                if dataset_weight <1.0 :
                    num_samples =int (len (train_examples )*dataset_weight )
                    import random 
                    train_examples =random .sample (train_examples ,num_samples )

                all_train_examples .extend (train_examples )
                print (f"Добавлено {len(train_examples)} примеров из {dataset_config['name']} (train, {len(dataset_config['languages'])} языков, weight={dataset_weight})")

            if dataset_config .get ("use_dev",False ):
                dev_examples =load_sts_dataset_multilang (
                dataset_config ["hf_repo"],
                dataset_config ["languages"],
                split ="dev"
                )
                all_train_examples .extend (dev_examples )
                print (f"Добавлено {len(dev_examples)} примеров из {dataset_config['name']} (dev)")

    if not all_train_examples :
        print ("Ошибка: не найдено примеров для обучения!")
        return 

    print (f"\nВсего примеров для обучения: {len(all_train_examples)}")
    print (f"Тестовых evaluators: {len(test_evaluators)}")
    print ()


    train_dataloader =DataLoader (
    all_train_examples ,
    shuffle =True ,
    batch_size =CONFIG ["training"]["batch_size"],
    pin_memory =True if torch .cuda .is_available ()else False ,
    )


    start_epoch =1 
    if args .resume_from :
        print (f"Загрузка модели из {args.resume_from}...")
        model =CrossEncoder (args .resume_from )

        try :
            start_epoch =int (args .resume_from .split ("epoch_")[-1 ])+1 
            print (f"Продолжаем с эпохи {start_epoch}")
        except :
            print ("Не удалось определить эпоху, начинаем с 1")
    else :
        base_model_path =CONFIG ["base_model"]
        print (f"Инициализация модели: {base_model_path}")


        if Path (base_model_path ).exists ():
            print (f"  → Используем предобученную модель (transfer learning)")
            print (f"  → Модель уже обучена на IR датасете, дообучаем на STS")
            model =CrossEncoder (base_model_path ,num_labels =1 )
        else :
            print (f"  → Начинаем обучение с нуля")
            model =CrossEncoder (base_model_path ,num_labels =1 )


        initial_checkpoint =checkpoint_dir /"initial_model"
        model .save (str (initial_checkpoint ))
        print (f"Сохранен начальный checkpoint: {initial_checkpoint}")

    start_time =time .time ()

    try :

        for epoch in range (start_epoch ,CONFIG ["training"]["num_epochs"]+1 ):
            print (f"\n{'='*80}")
            print (f"Эпоха {epoch}/{CONFIG['training']['num_epochs']}")
            print (f"{'='*80}\n")

            epoch_start_time =time .time ()


            eval_steps =CONFIG ["training"]["evaluation_steps"]if test_evaluators else None 
            evaluator =test_evaluators [0 ]if test_evaluators else None 

            model .fit (
            train_dataloader =train_dataloader ,
            epochs =1 ,
            warmup_steps =CONFIG ["training"]["warmup_steps"]if epoch ==start_epoch else 0 ,
            optimizer_params ={"lr":CONFIG ["training"]["learning_rate"]},
            evaluator =evaluator ,
            evaluation_steps =eval_steps ,
            output_path =str (output_dir ),
            save_best_model =True if evaluator else False ,
            use_amp =True ,
            show_progress_bar =True ,
            )


            epoch_checkpoint =checkpoint_dir /f"epoch_{epoch}"
            model .save (str (epoch_checkpoint ))
            print (f"\n✓ Checkpoint сохранен: {epoch_checkpoint}")

            epoch_elapsed =time .time ()-epoch_start_time 
            print (f"Время эпохи: {epoch_elapsed / 60:.1f} минут")


            progress ={
            "epoch":epoch ,
            "total_epochs":CONFIG ["training"]["num_epochs"],
            "elapsed_time":time .time ()-start_time ,
            "checkpoint_path":str (epoch_checkpoint ),
            }
            progress_path =output_dir /"training_progress.json"
            with progress_path .open ("w",encoding ="utf-8")as f :
                json .dump (progress ,f ,indent =2 ,ensure_ascii =False )

        elapsed_time =time .time ()-start_time 
        print (f"\n{'='*80}")
        print (f"Обучение завершено!")
        print (f"Время обучения: {elapsed_time / 3600:.2f} часов")
        print (f"{'='*80}\n")

    except KeyboardInterrupt :
        print ("\n\nОбучение прервано пользователем (Ctrl+C)")
        elapsed_time =time .time ()-start_time 
        print (f"Время обучения до прерывания: {elapsed_time / 3600:.2f} часов")

    except Exception as e :
        print (f"\n\n{'='*80}")
        print (f"Ошибка во время обучения: {e}")
        import traceback 
        traceback .print_exc ()
        print (f"{'='*80}\n")


        error_checkpoint =checkpoint_dir /"error_model"
        try :
            model .save (str (error_checkpoint ))
            print (f"✓ Модель сохранена в: {error_checkpoint}")
        except Exception as save_error :
            print (f"⚠ Не удалось сохранить модель: {save_error}")

        elapsed_time =time .time ()-start_time 
        print (f"Время обучения до ошибки: {elapsed_time / 3600:.2f} часов")
        raise 

    finally :

        final_model_path =output_dir /"final_model"
        try :
            model .save (str (final_model_path ))
            print (f"\n✓ Финальная модель сохранена в: {final_model_path}")
        except Exception as e :
            print (f"⚠ Предупреждение: не удалось сохранить финальную модель: {e}")


    if test_evaluators :
        print ("Финальная оценка на тестовых наборах:")
        for evaluator in test_evaluators :
            results =evaluator (model )
            print (f"\n{evaluator.name}:")
            for metric ,value in results .items ():
                print (f"  {metric}: {value:.4f}")


    config_path =output_dir /"training_config.json"
    with config_path .open ("w",encoding ="utf-8")as f :
        json .dump (CONFIG ,f ,indent =2 ,ensure_ascii =False )

    print (f"\nМодель сохранена в: {output_dir}")


if __name__ =="__main__":
    main ()

