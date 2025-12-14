from app.vector_search.db import *
from app.vector_search.utils import Tags
from app.vector_search.inference.embedder import TextEmbedder

import numpy as np


class SearchEngine:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.database = ...

    def search(self, tags: Tags):
        for _, tags in tags.items():
            np.mean(
            [
                self.text_embedder.process(tag_value)
                for tag_value in tags
            ],
            axis=0
            )


        # self.database.search()
