from datasets import load_dataset

ds = load_dataset("yandex/yambda", data_dir="flat/50m", data_files="multi_event.parquet")
ds.save_to_disk("yambda")
