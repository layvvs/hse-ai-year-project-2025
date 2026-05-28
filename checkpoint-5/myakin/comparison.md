## likes/dislikes | Without EDA

|Model name|NDCG@10|Recall@10|
|----|----|----|
Ours UserKnn|0.018|0.02|
Ours ItemKNN|0.004|0.003|
Ours iALS|0.016|0.02|
Ours EASE|0.019|0.02|
Y ItemKNN|0.013|0.02|
Y iALS|0.008|0.01|


## likes/dislikes | With EDA

### *Binary interactions matrix (-1, 1)* 

|Model name|NDCG@10|Recall@10|mAP@10|HitRate@10|
|----|----|----|----|----|
Ours UserKnn|0.018|**0.023**|**0.013**|**0.05**|
Ours ItemKNN|0.003|0.003|0.002|0.007|
Ours iALS|0.016|**0.02**|0.01|**0.05**|
Ours ELSA|**0.019**|**0.02**|0.01|**0.05**|
TopPopular|0.01|0.01|0.004|0.03|
TopPersonal|0.018|**0.02**|0.01|**0.05**|
TopPersonal+TopPopular|0.017|0.018|0.012|0.04|
Yandex ItemKNN|0.013|0.02|—|—|
Yandex iALS|0.008|0.01|—|—|

### *weighted interactions matrix*

|Model name|NDCG@10|Recall@10|mAP@10|HitRate@10|
|----|----|----|----|----|
Ours UserKnn|0.017|0.02|0.02|0.05|
Ours ItemKNN|0.004|0.003|0.003|0.008|
Ours iALS|0.017|0.02|0.01|0.06|
Ours ELSA|**0.019**|0.02|0.01|0.05|
TopPopular|0.006|0.01|0.003|0.025|
TopPersonal|0.017|0.02|0.01|0.05|
TopPersonal+TopPopular|0.016|0.018|0.01|0.04|
Yandex ItemKNN|0.013|0.02|—|—|
Yandex iALS|0.008|0.01|—|—|


## likes/dislikes/listens | With EDA

### *weighted interactions matrix*

|Model name                  |NDCG@10 |Recall@10|mAP@10 |HitRate@10|
|---------------------------|--------|---------|-------|----------|
|Ours UserKNN               |0.008   |0.0025   |0.0040 |0.038     |
|Ours ItemKNN               |-       |-        |-      |-         |
|Ours iALS                  |**0.013**|0.0041  |**0.0061**|0.063 |
|Ours ELSA                  |0.011   |**0.0043**|0.0047 |**0.070**|
|TopPopular                 |0.006   |0.0019   |0.0029 |0.029     |
|TopPersonal                |0.008   |0.0025   |0.0040 |0.038     |
|TopPersonal+TopPopular     |0.007   |0.0020   |0.0037 |0.032     |
|Yandex ItemKNN|0.013|0.02|—|—|
|Yandex iALS|0.008|0.01|—|—|

### *binary interactions matrix (0, 1) | Only positives | With EDA*

| Model name | NDCG@100 | Recall@100 | mAP@100 | HitRate@100 |
| :--- | :--- | :--- | :--- | :--- |
| Ours UserKNN | 0.0310 | 0.0415 | 0.0079 | 0.3505 |
| Ours ItemKNN | — | — | — | — |
| Ours iALS | 0.0166 | 0.0213 | 0.0042 | 0.2076 |
| **Ours ELSA** | **0.0369** | **0.0522** | **0.0088** | **0.4206** |
| TopPopular | 0.0209 | 0.0296 | 0.0049 | 0.2781 |
| TopPersonal | 0.0310 | 0.0415 | 0.0079 | 0.3505 |
| TopPersonal+TopPopular | 0.0274 | 0.0342 | 0.0075 | 0.3151 |
|Yandex ItemKNN|0.013|0.02|—|—|
|Yandex iALS|0.008|0.01|—|—|

