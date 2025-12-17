from app.vector_search.db import Database
from app.vector_search.utils import Tags
from app.vector_search.inference.embedder import TextEmbedder

import numpy as np


class SearchEngine:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.database = Database()

    async def search(self, tags: Tags):
        embedding = await self.text_embedder.process(tags)
        return await self.database.search(embedding)
