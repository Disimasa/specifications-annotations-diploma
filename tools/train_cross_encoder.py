import json 
import time 
from pathlib import Path 
from typing import List ,Tuple 

import torch 
from datasets import load_dataset 
from sentence_transformers import CrossEncoder ,InputExample 
from sentence_transformers .cross_encoder .evaluation import (
CEBinaryClassificationEvaluator ,
CERerankingEvaluator ,
)
from torch .utils .data import DataLoader 


CONFIG ={
"base_model":"DeepPavlov/rubert-base-cased",
"output_dir":"models/cross-encoder-rusbeir",
"datasets":[
{
"name":"sberquad-retrieval",
"hf_repo":"kngrg/sberquad-retrieval",
"hf_repo_qrels":"kngrg/sberquad-retrieval-qrels",
"use_train":True ,
"use_test":True ,
},








],
"training":{
"num_epochs":3 ,
"batch_size":16 ,
"learning_rate":2e-5 ,
"warmup_steps":1000 ,
"max_length":512 ,
"evaluation_steps":5000 ,
"gradient_accumulation_steps":1 ,
},
}


def load_rusbeir_dataset (hf_repo :str ,hf_repo_qrels :str ,split :str ="train"):
    print (f"Загрузка датасета {hf_repo} (split={split})...")


    try :
        corpus_ds =load_dataset (hf_repo ,"corpus",split =split if split !="test"else "test")
        queries_ds =load_dataset (hf_repo ,"queries",split =split if split !="test"else "test")
        qrels_ds =load_dataset (hf_repo_qrels ,split =split if split !="test"else "test")
    except Exception as e :

        print (f"Попытка загрузки без конфигов...")
        try :
            corpus_ds =load_dataset (hf_repo ,split =split if split !="test"else "test")
            queries_ds =corpus_ds 
            qrels_ds =load_dataset (hf_repo_qrels ,split =split if split !="test"else "test")
        except Exception as e2 :
            raise Exception (f"Не удалось загрузить датасет: {e}, {e2}")


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

    print (f"Загружено: {len(corpus)} документов, {len(queries)} запросов, {len(qrels)} релевантных пар")
    return corpus ,queries ,qrels 


def create_training_examples (
corpus :dict ,queries :dict ,qrels :dict ,num_negatives :int =1 
)->List [InputExample ]:
    examples =[]
    all_doc_ids =list (corpus .keys ())

    print ("Создание примеров для обучения...")
    for query_id ,relevant_docs in qrels .items ():
        if query_id not in queries :
            continue 

        query_text =queries [query_id ]


        for doc_id ,score in relevant_docs .items ():
            if doc_id not in corpus or score <=0 :
                continue 

            doc_text =corpus [doc_id ]["text"]
            examples .append (InputExample (texts =[query_text ,doc_text ],label =1.0 ))


        relevant_doc_ids =set (relevant_docs .keys ())
        negative_candidates =[d for d in all_doc_ids if d not in relevant_doc_ids ]


        import random 
        num_neg =min (num_negatives *len (relevant_docs ),len (negative_candidates ))
        negative_docs =random .sample (negative_candidates ,num_neg )

        for doc_id in negative_docs :
            doc_text =corpus [doc_id ]["text"]
            examples .append (InputExample (texts =[query_text ,doc_text ],label =0.0 ))

    print (f"Создано {len(examples)} примеров ({sum(1 for e in examples if e.label == 1.0)} positive, {sum(1 for e in examples if e.label == 0.0)} negative)")
    return examples 


def create_evaluation_data (corpus :dict ,queries :dict ,qrels :dict ):
    sentence_pairs =[]
    labels =[]

    for query_id ,relevant_docs in qrels .items ():
        if query_id not in queries :
            continue 

        query_text =queries [query_id ]


        for doc_id in corpus .keys ():
            doc_text =corpus [doc_id ]["text"]
            sentence_pairs .append ([query_text ,doc_text ])


            label =1.0 if doc_id in relevant_docs and relevant_docs [doc_id ]>0 else 0.0 
            labels .append (label )

    return sentence_pairs ,labels 


def main ():
    import argparse 

    parser =argparse .ArgumentParser (description ="Обучение cross-encoder для Information Retrieval")
    parser .add_argument (
    "--resume_from",
    type =str ,
    default =None ,
    help ="Путь к checkpoint для продолжения обучения (например: models/cross-encoder-rusbeir/checkpoints/epoch_2)",
    )
    args =parser .parse_args ()

    print ("="*80 )
    print ("Обучение Cross-Encoder для Information Retrieval")
    print ("="*80 )
    print (f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch .cuda .is_available ():
        print (f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print ()


    output_dir =Path (CONFIG ["output_dir"])
    output_dir .mkdir (parents =True ,exist_ok =True )


    all_train_examples =[]
    test_evaluators =[]

    for dataset_config in CONFIG ["datasets"]:
        name =dataset_config ["name"]
        print (f"\n{'='*80}")
        print (f"Обработка датасета: {name}")
        print (f"{'='*80}")


        if dataset_config ["use_train"]:
            try :
                corpus_train ,queries_train ,qrels_train =load_rusbeir_dataset (
                dataset_config ["hf_repo"],
                dataset_config ["hf_repo_qrels"],
                split ="train"
                )
                train_examples =create_training_examples (
                corpus_train ,queries_train ,qrels_train ,num_negatives =1 
                )
                all_train_examples .extend (train_examples )
                print (f"Добавлено {len(train_examples)} примеров из {name}")
            except Exception as e :
                print (f"Ошибка при загрузке train данных {name}: {e}")


        if dataset_config ["use_test"]:
            try :

                try :
                    corpus_test ,queries_test ,qrels_test =load_rusbeir_dataset (
                    dataset_config ["hf_repo"],
                    dataset_config ["hf_repo_qrels"],
                    split ="test"
                    )
                except :

                    print (f"  Test split не найден, используем часть train данных для валидации")
                    corpus_test =corpus_train 
                    queries_test =queries_train 
                    qrels_test =qrels_train 


                    query_ids =list (queries_test .keys ())[:1000 ]
                    queries_test ={qid :queries_test [qid ]for qid in query_ids }
                    qrels_test ={qid :qrels_test [qid ]for qid in query_ids if qid in qrels_test }


                sentence_pairs ,labels =create_evaluation_data (
                corpus_test ,queries_test ,qrels_test 
                )


                if len (sentence_pairs )>10000 :
                    import random 
                    indices =random .sample (range (len (sentence_pairs )),10000 )
                    sentence_pairs =[sentence_pairs [i ]for i in indices ]
                    labels =[labels [i ]for i in indices ]

                evaluator =CEBinaryClassificationEvaluator (
                sentence_pairs =sentence_pairs ,
                labels =labels ,
                name =name ,
                show_progress_bar =True ,
                )
                test_evaluators .append (evaluator )
                print (f"Создан evaluator для {name} ({len(sentence_pairs)} пар)")
            except Exception as e :
                print (f"Ошибка при загрузке test данных {name}: {e}")
                print (f"  Продолжаем без evaluator для этого датасета")

    if not all_train_examples :
        print ("Ошибка: не удалось загрузить обучающие данные!")
        return 

    print (f"\n{'='*80}")
    print (f"Всего примеров для обучения: {len(all_train_examples)}")
    print (f"Positive: {sum(1 for e in all_train_examples if e.label == 1.0)}")
    print (f"Negative: {sum(1 for e in all_train_examples if e.label == 0.0)}")
    print (f"{'='*80}\n")


    train_dataloader =DataLoader (
    all_train_examples ,
    shuffle =True ,
    batch_size =CONFIG ["training"]["batch_size"],
    )


    if args .resume_from :
        print (f"Продолжение обучения из checkpoint: {args.resume_from}")
        model =CrossEncoder (args .resume_from )


        progress_path =Path (args .resume_from ).parent .parent /"training_progress.json"
        if progress_path .exists ():
            with progress_path .open ("r",encoding ="utf-8")as f :
                progress =json .load (f )
                start_epoch =progress .get ("epoch",1 )+1 
                print (f"Продолжаем с эпохи {start_epoch}")
        else :
            start_epoch =1 
            print ("Информация о прогрессе не найдена, начинаем с эпохи 1")
    else :
        print (f"Инициализация новой модели: {CONFIG['base_model']}")
        model =CrossEncoder (
        CONFIG ["base_model"],
        num_labels =1 ,
        max_length =CONFIG ["training"]["max_length"],
        )
        start_epoch =1 


    num_batches =len (train_dataloader )
    num_epochs =CONFIG ["training"]["num_epochs"]
    total_steps =num_batches *num_epochs 

    print (f"\nПараметры обучения:")
    print (f"  Эпох: {num_epochs}")
    print (f"  Батчей на эпоху: {num_batches}")
    print (f"  Всего шагов: {total_steps}")
    print (f"  Batch size: {CONFIG['training']['batch_size']}")
    print (f"  Learning rate: {CONFIG['training']['learning_rate']}")
    print (f"  Оценка каждые: {CONFIG['training']['evaluation_steps']} шагов")



    estimated_time_per_batch =0.2 
    estimated_time_hours =(num_batches *num_epochs *estimated_time_per_batch )/3600 

    print (f"\nОценка времени обучения:")
    print (f"  Примерное время на батч: ~{estimated_time_per_batch:.1f} сек")
    print (f"  Примерное время на эпоху: ~{num_batches * estimated_time_per_batch / 3600:.1f} часов")
    print (f"  Примерное общее время: ~{estimated_time_hours:.1f} часов")
    print (f"  (Это приблизительная оценка, реальное время может отличаться)")


    print (f"\n{'='*80}")
    print ("Начало обучения...")
    print (f"{'='*80}\n")

    start_time =time .time ()


    checkpoint_dir =output_dir /"checkpoints"
    checkpoint_dir .mkdir (parents =True ,exist_ok =True )


    if not args .resume_from :
        initial_checkpoint =checkpoint_dir /"initial_model"
        model .save (str (initial_checkpoint ))
        print (f"Сохранен начальный checkpoint: {initial_checkpoint}")

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
            print (f"Время эпохи: {epoch_elapsed / 60:.1f} минут ({epoch_elapsed / 3600:.2f} часов)")


            progress_info ={
            "epoch":epoch ,
            "total_epochs":CONFIG ["training"]["num_epochs"],
            "elapsed_time":time .time ()-start_time ,
            "checkpoint_path":str (epoch_checkpoint ),
            }
            progress_path =output_dir /"training_progress.json"
            with progress_path .open ("w",encoding ="utf-8")as f :
                json .dump (progress_info ,f ,indent =2 ,ensure_ascii =False )

        elapsed_time =time .time ()-start_time 
        print (f"\n{'='*80}")
        print (f"Обучение успешно завершено!")
        print (f"Время обучения: {elapsed_time / 3600:.2f} часов ({elapsed_time / 60:.1f} минут)")
        print (f"{'='*80}\n")

    except KeyboardInterrupt :
        print (f"\n\n{'='*80}")
        print ("Обучение прервано пользователем (Ctrl+C)")
        print (f"{'='*80}\n")


        interrupted_checkpoint =checkpoint_dir /"interrupted_model"
        model .save (str (interrupted_checkpoint ))
        print (f"✓ Модель сохранена в: {interrupted_checkpoint}")
        print (f"  Для продолжения обучения используйте: --resume_from {interrupted_checkpoint}")

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

