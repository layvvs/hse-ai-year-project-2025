from datasets import load_dataset

ds = load_dataset("yandex/yambda", data_files="embeddings.parquet")
ds.save_to_disk("yambda_embedings")
