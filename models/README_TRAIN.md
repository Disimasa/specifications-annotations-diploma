## Обучение 40к сэмплов 32 батч
python scripts/train/finetune_bi_encoder.py --epochs 2 --batch-size 32 --learning-rate 1e-5 --max-train-samples 40000

  test_gisnauka: R@20=0.4693  P@20=0.0762  MRR@20=0.3716  n=63
  test_gisnauka_doc_level: R@20=0.4704  P@20=0.0770  MRR@20=0.4270  n=63
  test_gisnauka_docs: R@20=0.5553  P@20=0.0691  MRR@20=0.3642  n=709
  test_gisnauka_docs_doc_level: R@20=0.5406  P@20=0.0673  MRR@20=0.3629  n=709
  gold_jsonl: R@20=0.4463  P@20=0.0833  MRR@20=0.3214  n=9
  gold_jsonl_doc_level: R@20=0.4019  P@20=0.0722  MRR@20=0.3537  n=9
  valid: R@20=0.5118  P@20=0.0671  MRR@20=0.3267  n=1419
  valid_doc_level: R@20=0.5109  P@20=0.0662  MRR@20=0.3271  n=1419

## Полное обучение 230k сэмплов 32 батч в несколько шагов DEPRECATED

1. Шаг 1 40к сэмплов
python scripts/train/finetune_bi_encoder.py --output-dir models/bi-encoder-gisnauka-200k --epochs 2 --batch-size 32 --learning-rate 1e-5 --max-train-samples 40000
test_gisnauka: R@20=0.4733  P@20=0.0770  MRR@20=0.3819  n=63
test_gisnauka_doc_level: R@20=0.4423  P@20=0.0722  MRR@20=0.4226  n=63
  test_gisnauka_docs: R@20=0.5486  P@20=0.0683  MRR@20=0.3677  n=709
  test_gisnauka_docs_doc_level: R@20=0.5449  P@20=0.0680  MRR@20=0.3675  n=709
  gold_jsonl: R@20=0.4463  P@20=0.0833  MRR@20=0.3171  n=9
  gold_jsonl_doc_level: R@20=0.3833  P@20=0.0667  MRR@20=0.3196  n=9
  valid: R@20=0.5127  P@20=0.0671  MRR@20=0.3259  n=1419
  valid_doc_level: R@20=0.5097  P@20=0.0665  MRR@20=0.3308  n=1419


2. Шаг 2 40к сэмплов
python scripts/train/finetune_bi_encoder.py --resume models/bi-encoder-gisnauka-200k --output-dir models/bi-encoder-gisnauka-200k --epochs 2 --batch-size 32 --learning-rate 1e-5 --max-train-samples 40000

## Обучение на полной выборке батч 128 1 эпоха (записано в папку bi-encoder-gisnauka-full-64)
```shell
python scripts/train/finetune_bi_encoder.py --output-dir models/bi-encoder-gisnauka-full-64 --epochs 1 --batch-size 128 --learning-rate 1e-5 --save-steps 500
```

--- Summary ---
  test_gisnauka: R@20=0.5122  P@20=0.0833  MRR@20=0.3855  n=63
  test_gisnauka_doc_level: R@20=0.4934  P@20=0.0810  MRR@20=0.4297  n=63
  test_gisnauka_docs: R@20=0.5587  P@20=0.0713  MRR@20=0.3918  n=709
  test_gisnauka_docs_doc_level: R@20=0.5739  P@20=0.0714  MRR@20=0.3964  n=709
  gold_jsonl: R@20=0.4593  P@20=0.0889  MRR@20=0.3463  n=9
  gold_jsonl_doc_level: R@20=0.3278  P@20=0.0556  MRR@20=0.2966  n=9
  valid: R@20=0.5347  P@20=0.0700  MRR@20=0.3498  n=1419
  valid_doc_level: R@20=0.5383  P@20=0.0697  MRR@20=0.3508  n=1419

2 эпоха
```shell
python scripts/train/finetune_bi_encoder.py --resume models/bi-encoder-gisnauka-full-64 --output-dir models/bi-encoder-gisnauka-full-128 --epochs 2 --batch-size 128 --learning-rate 1e-5 --save-steps 500
```

--- Summary ---
  test_gisnauka: R@20=0.5177  P@20=0.0833  MRR@20=0.4075  n=63
  test_gisnauka_doc_level: R@20=0.5024  P@20=0.0810  MRR@20=0.4214  n=63
  test_gisnauka_docs: R@20=0.5729  P@20=0.0724  MRR@20=0.3986  n=709
  test_gisnauka_docs_doc_level: R@20=0.5774  P@20=0.0720  MRR@20=0.3980  n=709
  gold_jsonl: R@20=0.3907  P@20=0.0722  MRR@20=0.3857  n=9
  gold_jsonl_doc_level: R@20=0.3796  P@20=0.0611  MRR@20=0.2830  n=9
  valid: R@20=0.5438  P@20=0.0731  MRR@20=0.3586  n=1419
  valid_doc_level: R@20=0.5501  P@20=0.0715  MRR@20=0.3605  n=1419

!Сильно хуже на JSONL test

## Обучение на полной выборке батч 1024 1 эпоха
```shell
python scripts/train/finetune_bi_encoder.py --output-dir models/bi-encoder-gisnauka-full-1024 --epochs 1 --batch-size 1024 --learning-rate 1e-5 --save-steps 500
```
Обучение 20 часов на одной эпохе!!!

## обучение на полной выборке батч 256 1 эпоха
python scripts/train/finetune_bi_encoder.py --output-dir models/bi-encoder-gisnauka-full-256 --epochs 1 --batch-size 256 --learning-rate 1e-5 --save-steps 200

test_gisnauka: R@20=0.4852  P@20=0.0794  MRR@20=0.3932  R@M=0.1762  P@M=0.1762  n=63
test_gisnauka_doc_level: R@20=0.4847  P@20=0.0802  MRR@20=0.4396  R@M=0.1884  P@M=0.1884  n=63
test_gisnauka_docs: R@20=0.5646  P@20=0.0715  MRR@20=0.3920  R@M=0.2035  P@M=0.2040  n=709
test_gisnauka_docs_doc_level: R@20=0.5641  P@20=0.0702  MRR@20=0.3951  R@M=0.2113  P@M=0.2113  n=709
gold_jsonl: R@20=0.4648  P@20=0.0889  MRR@20=0.4208  R@M=0.2148  P@M=0.2148  n=9
gold_jsonl_doc_level: R@20=0.3833  P@20=0.0667  MRR@20=0.3203  R@M=0.0963  P@M=0.0963  n=9
valid: R@20=0.5274  P@20=0.0692  MRR@20=0.3519  R@M=0.1887  P@M=0.1887  n=1419
valid_doc_level: R@20=0.5282  P@20=0.0686  MRR@20=0.3476  R@M=0.1899  P@M=0.1899  n=1419

## TODO: добавить путь к предобученным батчам. Обучение на полной выборке батч 128 с кастомным батчером 1 эпоха
Если запускать с precomputed
```shell
python scripts/train/finetune_bi_encoder.py --output-dir=models/bi-encoder-gisnauka-full-256-batcher --precomputed-batches data/gold/precomputed_batches/hb_....pt --batch-size=128 --loss gist --gist-relative-margin 0.05 --curriculum-epoch1 "0.8,0.2,0" --curriculum-epoch2 "0.6,0.3,0.1" --curriculum-epoch3plus "0.45,0.35,0.2" --epochs 1
```
Без guide -
```shell
python scripts/train/finetune_bi_encoder.py --output-dir=models/bi-encoder-gisnauka-full-256-batcher --use-hierarchical-sampler --disable-guide-safe-hard --loss cached_mnr --epochs 1 --batch-size 128 --curriculum-epoch1 "0.8,0.2,0"
```

## TODO: Предгенерация батчей
```shell
python scripts\train\generate_hierarchical_batches.py --output-dir data\gold\precomputed_batches --epochs 1 --batch-size 128 --seed 42 --disable-guide-safe-hard --curriculum-epoch1 "0.8,0.2,0"
```

python scripts\train\playground_batcher_quality.py

Сравнение на 10к с разными батчерами
```shell
python scripts/train/finetune_bi_encoder.py --output-dir models/compare-no-dup-10k --max-train-samples 10000 --seed 42 --epochs 1 --batch-size 128 --mini-batch-size 32 --learning-rate 1e-5 --loss cached_mnr --save-steps 500
```
  test_gisnauka: R@20=0.4397  P@20=0.0738  MRR@20=0.4191  R@M=0.1796  P@M=0.1796  n=63
  test_gisnauka_doc_level: R@20=0.4429  P@20=0.0714  MRR@20=0.3760  R@M=0.2082  P@M=0.2082  n=63
  test_gisnauka_docs: R@20=0.4805  P@20=0.0608  MRR@20=0.3469  R@M=0.1784  P@M=0.1784  n=709
  test_gisnauka_docs_doc_level: R@20=0.4995  P@20=0.0618  MRR@20=0.3536  R@M=0.1838  P@M=0.1838  n=709
  gold_jsonl: R@20=0.4426  P@20=0.0778  MRR@20=0.3506  R@M=0.2074  P@M=0.2074  n=9
  gold_jsonl_doc_level: R@20=0.3333  P@20=0.0556  MRR@20=0.3287  R@M=0.1019  P@M=0.1019  n=9
  valid: R@20=0.4612  P@20=0.0599  MRR@20=0.2919  R@M=0.1513  P@M=0.1513  n=1419
  valid_doc_level: R@20=0.4708  P@20=0.0609  MRR@20=0.3048  R@M=0.1561  P@M=0.1561  n=1419

С иерархичным батчером 
```text
[I 2026-03-22 05:36:00,422] Trial 6 finished with value: 0.4674603174603174 and parameters: {'max_scored_candidates': 256, 'leaf_balance_power': 0.8, 'grand_balance_weight': 0.8, 'curriculum_epoch1': '0,0,1'}. Best is trial 6 with value: 0.4674603174603174.
```
 test_gisnauka: R@20=0.4675  P@20=0.0762  MRR@20=0.4211  R@M=0.1899  P@M=0.1899  n=63
  test_gisnauka_doc_level: R@20=0.4466  P@20=0.0730  MRR@20=0.4185  R@M=0.1886  P@M=0.1886  n=63
  test_gisnauka_docs: R@20=0.4802  P@20=0.0598  MRR@20=0.3400  R@M=0.1758  P@M=0.1758  n=709
  test_gisnauka_docs_doc_level: R@20=0.5004  P@20=0.0618  MRR@20=0.3562  R@M=0.1886  P@M=0.1886  n=709
  gold_jsonl: R@20=0.4648  P@20=0.0833  MRR@20=0.4432  R@M=0.2352  P@M=0.2352  n=9
  gold_jsonl_doc_level: R@20=0.4074  P@20=0.0722  MRR@20=0.3014  R@M=0.1296  P@M=0.1296  n=9
  valid: R@20=0.4619  P@20=0.0602  MRR@20=0.2975  R@M=0.1544  P@M=0.1544  n=1419
  valid_doc_level: R@20=0.4684  P@20=0.0603  MRR@20=0.3025  R@M=0.1563  P@M=0.1563  n=1419


## сгенерил батчи 128 для обучения
```shell
python scripts\train\generate_hierarchical_batches.py --batch-size 128 --seed 42 --epochs 1 --relative-margin 0.05 --curriculum-epoch1 "0,0,1" --leaf-balance-power 0.8 --grand-balance-weight 0.8 --max-scored-candidates 256 --disable-guide-safe-hard --disable-sampler-diagnostics
```

## обучение на сгенеренных батчах (БАГ в обучении???)
```shell
python scripts\train\finetune_bi_encoder.py --output-dir models\bi-encoder-full-128-batcher --epochs 1 --batch-size 128 --loss cached_mnr --precomputed-batches "data\gold\precomputed_batches\hb_gisnauka_segments_train_augmented_bs128_ep1_seed42_m0.05_guide-user-bge-m3_c1-0-0-1_c2-0.6-0.3-0.1_c3-0.45-0.35-0.2_54435d722b.pt" --skip-baseline-test
```

test_gisnauka: R@20=0.5135  P@20=0.0833  MRR@20=0.4092  R@M=0.1672  P@M=0.1672  n=63
  test_gisnauka_doc_level: R@20=0.5132  P@20=0.0833  MRR@20=0.3757  R@M=0.1772  P@M=0.1772  n=63
  test_gisnauka_docs: R@20=0.5579  P@20=0.0692  MRR@20=0.3758  R@M=0.2029  P@M=0.2029  n=709
  test_gisnauka_docs_doc_level: R@20=0.5494  P@20=0.0688  MRR@20=0.3808  R@M=0.2009  P@M=0.2009  n=709
  gold_jsonl: R@20=0.4741  P@20=0.0889  MRR@20=0.2861  R@M=0.1648  P@M=0.1648  n=9
  gold_jsonl_doc_level: R@20=0.4074  P@20=0.0722  MRR@20=0.2984  R@M=0.0463  P@M=0.0463  n=9
  valid: R@20=0.5171  P@20=0.0677  MRR@20=0.3428  R@M=0.1827  P@M=0.1827  n=1419
  valid_doc_level: R@20=0.5237  P@20=0.0680  MRR@20=0.3425  R@M=0.1844  P@M=0.1844  n=1419

## TODO: повторить обучение после фикса кода обучения
```shell
python scripts\train\finetune_bi_encoder.py --output-dir models\bi-encoder-full-128-batcher-fixed --epochs 1 --batch-size 128 --loss cached_mnr --precomputed-batches "data\gold\precomputed_batches\hb_gisnauka_segments_train_augmented_bs128_ep1_seed42_m0.05_guide-user-bge-m3_c1-0-0-1_c2-0.6-0.3-0.1_c3-0.45-0.35-0.2_54435d722b.pt" --skip-baseline-test
```
  test_gisnauka: R@20=0.5095  P@20=0.0833  MRR@20=0.3901  R@M=0.1749  P@M=0.1749  n=63
  test_gisnauka_doc_level: R@20=0.4910  P@20=0.0786  MRR@20=0.4159  R@M=0.1971  P@M=0.1971  n=63
  test_gisnauka_docs: R@20=0.5544  P@20=0.0693  MRR@20=0.3863  R@M=0.2070  P@M=0.2075  n=709
  test_gisnauka_docs_doc_level: R@20=0.5461  P@20=0.0686  MRR@20=0.3872  R@M=0.2009  P@M=0.2009  n=709
  gold_jsonl: R@20=0.4685  P@20=0.0889  MRR@20=0.3593  R@M=0.2241  P@M=0.2241  n=9
  gold_jsonl_doc_level: R@20=0.2963  P@20=0.0500  MRR@20=0.2702  R@M=0.0463  P@M=0.0463  n=9
  valid: R@20=0.5229  P@20=0.0687  MRR@20=0.3437  R@M=0.1820  P@M=0.1820  n=1419
  valid_doc_level: R@20=0.5264  P@20=0.0685  MRR@20=0.3481  R@M=0.1901  P@M=0.1901  n=1419
  
### checkpoint 1500
  test_gisnauka: R@20=0.5116  P@20=0.0849  MRR@20=0.3959  R@M=0.1701  P@M=0.1701  n=63
  test_gisnauka_doc_level: R@20=0.5056  P@20=0.0833  MRR@20=0.4194  R@M=0.2066  P@M=0.2066  n=63
  test_gisnauka_docs: R@20=0.5581  P@20=0.0712  MRR@20=0.4023  R@M=0.2149  P@M=0.2149  n=709
  test_gisnauka_docs_doc_level: R@20=0.5619  P@20=0.0707  MRR@20=0.4073  R@M=0.2164  P@M=0.2164  n=709
  gold_jsonl: R@20=0.4870  P@20=0.0944  MRR@20=0.4548  R@M=0.1870  P@M=0.1870  n=9
  gold_jsonl_doc_level: R@20=0.3648  P@20=0.0667  MRR@20=0.3153  R@M=0.0833  P@M=0.0833  n=9
  valid: R@20=0.5342  P@20=0.0706  MRR@20=0.3546  R@M=0.1907  P@M=0.1907  n=1419
  valid_doc_level: R@20=0.5392  P@20=0.0700  MRR@20=0.3618  R@M=0.1941  P@M=0.1941  n=1419
Тут проблема в том, что последние 35% батчей очень маленькие из-за ограничений в генерации батчей

## Генерация батчей с relaxed ограничениями
```shell
python scripts\train\generate_hierarchical_batches.py --batch-size 128 --seed 42 --epochs 1 --relative-margin 0.05 --c
urriculum-epoch1 "0,0,1" --curriculum-epoch2 "0.6,0.3,0.1" --curriculum-epoch3plus "0.45,0.35,0.2" --leaf-balance-power 0.8 --grand-balance-weight 0.8 --max-scored-candidates 
256 --disable-guide-safe-hard --disable-sampler-diagnostics --output-dir "data\gold\precomputed_batches" --output-name "hb_gisnauka_smartpad_fixed_bs128_ep1_seed42_m0.05_c1-0-0-1_c2-0.6-0.3-0.1_c3-0.45-0.35-0.2_lb0.8_gw0.8.pt"
```

## TODO: обучение на батчах с нормальным tail
```shell
python scripts\train\finetune_bi_encoder.py --output-dir models\bi-encoder-precomputed-128-fnfilter-001 --epochs 1 --batch-size 128 --loss cached_mnr --precomputed-batches "data\gold\precomputed_batches\hb_gisnauka_smartpad_fixed_bs128_ep1_seed42_m0.05_c1-0-0-1_c2-0.6-0.3-0.1_c3-0.45-0.35-0.2_lb0.8_gw0.8.pt" --skip-baseline-test --filter-fn-pair-frac-max 0.01
```
  test_gisnauka: R@20=0.5272  P@20=0.0873  MRR@20=0.4081  R@M=0.1741  P@M=0.1741  n=63
  test_gisnauka_doc_level: R@20=0.5124  P@20=0.0833  MRR@20=0.4149  R@M=0.2225  P@M=0.2225  n=63
  test_gisnauka_docs: R@20=0.5597  P@20=0.0706  MRR@20=0.4022  R@M=0.2068  P@M=0.2068  n=709
  test_gisnauka_docs_doc_level: R@20=0.5680  P@20=0.0712  MRR@20=0.4030  R@M=0.2108  P@M=0.2108  n=709
  gold_jsonl: R@20=0.4370  P@20=0.0833  MRR@20=0.4605  R@M=0.1593  P@M=0.1593  n=9
  gold_jsonl_doc_level: R@20=0.3463  P@20=0.0611  MRR@20=0.3343  R@M=0.1333  P@M=0.1333  n=9
  valid: R@20=0.5368  P@20=0.0712  MRR@20=0.3558  R@M=0.1917  P@M=0.1923  n=1419
  valid_doc_level: R@20=0.5386  P@20=0.0699  MRR@20=0.3578  R@M=0.1934  P@M=0.1934  n=1419
  
### Распределение хард негативов
same_parent: 23.59% (210,603) — mean similarity 0.6433
same_grand_diff_parent: 30.48% (272,181) — mean similarity 0.6218
diff_grand: 45.93% (410,162) — mean similarity 0.5719

## Хард негатив майнинг
```shell
python scripts\train\mine_hard_negatives_segments.py --model "models\bi-encoder-precomputed-128-fnfilter-001" --ontology-embeddings "data\ontology_grnti_embeddings_fnfilter001.npz" --segments-csv "data\gold\gisnauka_segments_train_augmented.csv" --out-jsonl "data\gold\hard_negatives_train_fnfilter001.jsonl" --out-csv "data\gold\hard_negatives_train_fnfilter001.csv" --save-segment-embeddings "data\gold\segment_embeddings_fnfilter001.npz"
```

## Сборка триплетов
```shell
python scripts\train\build_triplets_from_hard_negatives.py --hard-jsonl data\gold\hard_negatives_train_fnfilter001.jsonl --out data\gold\triplets_train_fnfilter001.jsonl
```

## повторное дообучение на триплетах хард негатив
```shell
python scripts\train\finetune_bi_encoder.py --base-model models\bi-encoder-precomputed-128-fnfilter-001 --output-dir models\bi-encoder-triplet-fnfilter001 --loss triplet --triplets-jsonl data\gold\triplets_train_fnfilter001.jsonl --epochs 1 --batch-size 32 --learning-rate 2e-6 --triplet-margin 0.15 --skip-baseline-test
```
checkpoint 1700
test_gisnauka: R@20=0.0902  P@20=0.0151  MRR@20=0.0781  R@M=0.0317  P@M=0.0317  n=63
  test_gisnauka_doc_level: R@20=0.1119  P@20=0.0190  MRR@20=0.0890  R@M=0.0442  P@M=0.0442  n=63
  test_gisnauka_docs: R@20=0.0644  P@20=0.0088  MRR@20=0.0418  R@M=0.0161  P@M=0.0161  n=709
  test_gisnauka_docs_doc_level: R@20=0.0849  P@20=0.0113  MRR@20=0.0510  R@M=0.0221  P@M=0.0221  n=709
  gold_jsonl: R@20=0.0222  P@20=0.0056  MRR@20=0.0085  R@M=0.0000  P@M=0.0000  n=9
  gold_jsonl_doc_level: R@20=0.0000  P@20=0.0000  MRR@20=0.0000  R@M=0.0000  P@M=0.0000  n=9
  valid: R@20=0.0626  P@20=0.0081  MRR@20=0.0406  R@M=0.0153  P@M=0.0153  n=1419
  valid_doc_level: R@20=0.0804  P@20=0.0106  MRR@20=0.0520  R@M=0.0210  P@M=0.0210  n=1419

## Составляем батчи с хард негативами для cmnr
```shell
python scripts\train\generate_hard_negative_batches.py --jsonl data\gold\hard_negatives_train_fnfilter001.jsonl --segments-csv data\gold\gisnauka_segments_train_augmented.csv --batch-size 128 --epochs 1 --seed 42 --target-batches-per-epoch 2500 --hard-max-similarity 0.72 --hard-min-similarity 0.45
```

## Дообучение на составленых батчах с хард негативами
```shell
python scripts\train\finetune_bi_encoder.py --base-model models\bi-encoder-precomputed-128-fnfilter-001 --output-dir models\bi-encoder-cmnr-hardneg-mix-fnfilter001 --loss cached_mnr --precomputed-batches "data\gold\precomputed_batches\hb_hardneg_mix_hard_negatives_train_fnfilter001_bs128_ep1_seed42_0d483ed441.pt" --epochs 1 --batch-size 128 --learning-rate 2e-6 --seed 42 --skip-baseline-test
```
--- Summary ---
  test_gisnauka: R@20=0.5349  P@20=0.0873  MRR@20=0.3857  R@M=0.1767  P@M=0.1767  n=63
  test_gisnauka_doc_level: R@20=0.4881  P@20=0.0778  MRR@20=0.4198  R@M=0.2267  P@M=0.2267  n=63
  test_gisnauka_docs: R@20=0.5626  P@20=0.0722  MRR@20=0.4023  R@M=0.2102  P@M=0.2105  n=709
  test_gisnauka_docs_doc_level: R@20=0.5787  P@20=0.0723  MRR@20=0.4009  R@M=0.2122  P@M=0.2122  n=709
  gold_jsonl: R@20=0.4370  P@20=0.0833  MRR@20=0.4881  R@M=0.2148  P@M=0.2148  n=9
  gold_jsonl_doc_level: R@20=0.3741  P@20=0.0667  MRR@20=0.3121  R@M=0.0963  P@M=0.0963  n=9
  valid: R@20=0.5355  P@20=0.0728  MRR@20=0.3638  R@M=0.1908  P@M=0.1908  n=1419
  valid_doc_level: R@20=0.5414  P@20=0.0703  MRR@20=0.3628  R@M=0.1966  P@M=0.1966  n=1419

## Новые батчи с распределением по разным grandparent
```shell
python scripts\train\generate_hierarchical_batches.py --output-dir data\gold\precomputed_batches --epochs 1 --batch-size 128 --seed 42 --relative-margin 0.05 --curriculum-epoch1 "1,0,0" --curriculum-epoch2 "0.6,0.3,0.1" --curriculum-epoch3plus "0.45,0.35,0.2" --leaf-balance-power 0.8 --grand-balance-weight 0.8 --max-scored-candidates 256 --disable-guide-safe-hard --disable-sampler-diagnostics --output-name hb_grandfocus1-0-0_bs128_ep1_seed42_lb0.8_gw0.8.pt
```

##обучение с батчи с распределением по разным grandparent
```shell
python scripts\train\finetune_bi_encoder.py --base-model deepvk/USER-bge-m3 --output-dir models\bi-encoder-grandfocus100-from-vk --epochs 1 --batch-size 128 --loss cached_mnr --precomputed-batches "data\gold\precomputed_batches\hb_grandfocus1-0-0_bs128_ep1_seed42_lb0.8_gw0.8.pt" --learning-rate 1e-5 --seed 42 --skip-baseline-test --filter-fn-pair-frac-max 0.01
```
test_gisnauka: R@20=0.5140  P@20=0.0841  MRR@20=0.3818  R@M=0.1754  P@M=0.1754  n=63
  test_gisnauka_doc_level: R@20=0.5085  P@20=0.0833  MRR@20=0.4252  R@M=0.2312  P@M=0.2312  n=63
  test_gisnauka_docs: R@20=0.5701 P@20=0.0734  MRR@20=0.4057  R@M=0.2112  P@M=0.2125  n=709
  test_gisnauka_docs_doc_level: R@20=0.5693  P@20=0.0709  MRR@20=0.3993  R@M=0.2134  P@M=0.2134  n=709
  gold_jsonl: R@20=0.4593  P@20=0.0889  MRR@20=0.3685  R@M=0.1926  P@M=0.1926  n=9
  gold_jsonl_doc_level: R@20=0.4111  P@20=0.0722  MRR@20=0.3281  R@M=0.0870  P@M=0.0870  n=9
  valid: R@20=0.5334  P@20=0.0699  MRR@20=0.3529  R@M=0.1911  P@M=0.1911  n=1419
  valid_doc_level: R@20=0.5451  P@20=0.0706  MRR@20=0.3543  R@M=0.1921  P@M=0.1921  n=1419

## С новой онтологией 1 эпоха
```shell
python scripts\train\run_base_model_training_pipeline.py --ontology data\ontology_grnti_with_yagpt.json --mini-batch-size 4
```
--- Test gisnauka (полный пайплайн): n=63, skipped=0 ---
  R@20: 0.5209  P@20: 0.0857  MRR@20: 0.3711  R@M: 0.1902  P@M: 0.1902
Test gisnauka (document-level):
--- Test gisnauka (document-level): n=63 ---
  R@20: 0.4844  P@20: 0.0786  MRR@20: 0.4021  R@M: 0.1942  P@M: 0.1942
Test gisnauka docs (полный пайплайн):

--- Test gisnauka docs (полный пайплайн): n=709, skipped=0 ---
  R@20: 0.5685  P@20: 0.0731  MRR@20: 0.3967  R@M: 0.2136  P@M: 0.2144
Test gisnauka docs (document-level):
--- Test gisnauka docs (document-level): n=709 ---
  R@20: 0.5704  P@20: 0.0714  MRR@20: 0.4082  R@M: 0.2214  P@M: 0.2214


## Онтология YaGPT 1.5 эпохи (машина упала)
```shell
python scripts\train\run_base_model_training_pipeline.py --ontology data\ontology_grnti_with_yagpt.json --mini-batch-size 4 --epochs 3
```
  test_gisnauka: R@20=0.5093  P@20=0.0841  MRR@20=0.3872  R@M=0.1944  P@M=0.1944  n=63
  test_gisnauka_doc_level: R@20=0.5283  P@20=0.0857  MRR@20=0.4060  R@M=0.2249  P@M=0.2249  n=63
  test_gisnauka_docs: R@20=0.5642  P@20=0.0791  MRR@20=0.4203  R@M=0.2242  P@M=0.2261  n=709
  test_gisnauka_docs_doc_level: R@20=0.5838  P@20=0.0728  MRR@20=0.4186  R@M=0.2288  P@M=0.2288  n=709
  valid: R@20=0.5377  P@20=0.0769  MRR@20=0.3845  R@M=0.2058  P@M=0.2068  n=1419
  valid_doc_level: R@20=0.5486  P@20=0.0716  MRR@20=0.3839  R@M=0.2046  P@M=0.2046  n=1419