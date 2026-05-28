from app.vector_search.db import Database
from app.models.models import ForwardRequest
from app.vector_search.inference.embedder import TextEmbedder


class SearchEngine:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.database = Database()

    async def search(self, tags: ForwardRequest):
        embedding = await self.text_embedder.process(tags)
        return await self.database.search(embedding)
