import pickle
from pathlib import Path
from time import time


class ColdStartModel:
    def __init__(
            self,
            model_dir: str = "model-cold-start",
            min_interactions: int = 5,
    ):
        model_dir = Path(model_dir)

        with open(model_dir / "cluster_top_items.pkl", "rb") as f:
            self.cluster_top_items = pickle.load(f)

        with open(model_dir / "ordered_clusters.pkl", "rb") as f:
            self.ordered_clusters = pickle.load(f)

        self.min_interactions = min_interactions
        self.user_events = {}
        self.users_for_retrain = set()

    def add_event(
            self,
            uid: int,
            item_id: int,
            event_type: str,
            played_ratio_pct: int | None = None,
            track_length_seconds: int | None = None,
            is_organic: bool = False,
    ):
        if uid not in self.user_events:
            self.user_events[uid] = []

        self.user_events[uid].append(
            {
                "uid": uid,
                "item_id": item_id,
                "timestamp": int(time()),
                "is_organic": is_organic,
                "event_type": event_type,
                "played_ratio_pct": played_ratio_pct,
                "track_length_seconds": track_length_seconds,
            }
        )

        if len(self.user_events[uid]) == self.min_interactions:
            self.users_for_retrain.add(uid)
            # self._retrain_for_user(uid)

    def recommend(self, uid: int, k: int = 10) -> list[dict]:
        if uid not in self.user_events:
            self.user_events[uid] = []

        seen_items = {
            event["item_id"]
            for event in self.user_events[uid]
        }

        recs = []
        used = set()
        ptr = {cluster: 0 for cluster in self.ordered_clusters}

        while len(recs) < k:
            added = False

            for cluster in self.ordered_clusters:
                items = self.cluster_top_items[cluster]
                i = ptr[cluster]

                while i < len(items) and (items[i] in seen_items or items[i] in used):
                    i += 1

                ptr[cluster] = i

                if i >= len(items):
                    continue

                item_id = items[i]

                recs.append(
                    {
                        "item_id": int(item_id),
                        "score": 0.0,
                        "rank": len(recs) + 1,
                    }
                )

                used.add(item_id)
                ptr[cluster] += 1
                added = True

                if len(recs) == k:
                    break

            if not added:
                break

        return recs

    # TODO запуск дообучения
    def _retrain_for_user(self, uid: int):
        ...


ColdStartModel()