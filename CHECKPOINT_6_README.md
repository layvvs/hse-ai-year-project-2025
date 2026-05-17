## Что было сделано
* Обучены Item2Vec эмбеддинги треков (Word2Vec).
* Реализован ANN retrieval через NearestNeighbors.
* Построен baseline Item2Vec + ANN.
* Собраны признаки для reranking:

  * ann_rank
  * item_popularity
  * item_like_ratio
  * user_activity
* Обучен HistGradientBoostingClassifier с подбором гиперпараметров.
* Реализован neural reranker на MLPClassifier.
* Построены feature importance и графики сравнения моделей.

## Результаты
Item2Vec + ANN
* Recall@10 = 0.003065
* NDCG@10 = 0.003010
* MAP@10 = 0.002413

Item2Vec + ANN + Boosting
* Recall@10 = 0.008046
* NDCG@10 = 0.007265
* MAP@10 = 0.005392

Item2Vec + ANN + MLP
* Recall@10 = 0.004981
* NDCG@10 = 0.002529
* MAP@10 = 0.001091

## Вывод
Лучший результат показала модель Item2Vec + ANN + Boosting.
Boosting заметно улучшил baseline по ranking-метрикам, MLP оказался слабее.