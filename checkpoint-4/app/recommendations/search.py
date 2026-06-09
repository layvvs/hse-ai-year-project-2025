import asyncio

from app.models.models import ForwardRequest, SearchResultsList, SearchResult
from app.recommendations.engine import DEFAULT_K, load_engine


class SearchEngine:
    def __init__(self):
        self.engine = load_engine()

    async def search(self, params: ForwardRequest) -> SearchResultsList:
        k = params.k or DEFAULT_K
        loop = asyncio.get_running_loop()
        recommendations = await loop.run_in_executor(
            None,
            lambda: self.engine.recommend(params.uid, k=k),
        )
        return SearchResultsList(
            search_results=[
                SearchResult(
                    item_id=rec["item_id"],
                    score=rec["score"],
                    rank=rec["rank"],
                )
                for rec in recommendations
            ]
        )
