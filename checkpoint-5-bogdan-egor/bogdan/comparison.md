# likes/dislikes

##  No EDA

|Model name|NDCG@10|Recall@10|
|----|----|----|
Ours UserKnn|0.018|0.02|
Ours ItemKNN|0.004|0.003|
Ours iALS|0.016|0.02|
Ours EASE|0.019|0.02|
Y ItemKNN|0.013|0.02|
Y iALS|0.008|0.01|


## EDA

### sign matrix

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

### weighted matrix

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