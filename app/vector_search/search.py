from db import *

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import TypeAlias, Literal, Dict


TagName: TypeAlias = Literal['genre', 'instuments', 'vartags']
TagValues: TypeAlias = list[str]
Tags: TypeAlias = Dict[TagName, TagValues]


class SearchEngine:
    def __init__(self):
        self.text_embedder = SentenceTransformer('./src/models')
        self.database = ...

    def search(self, tags: Tags):
        for _, tags in tags.items():
            np.mean(
            [
                self.text_embedder.encode(tag_value)
                for tag_value in tags
            ],
            axis=0
            )


        # self.database.search()
