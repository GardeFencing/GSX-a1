import json
from typing import Any

import redis

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def cache_results(cache_key: str, data: Any) -> None:
    if isinstance(data, dict):
        redis_client.setex(cache_key, 86400, json.dumps(data))
    else:
        redis_client.setex(cache_key, 86400, data)


with open("redis_dump.json", "r") as file:
    data = json.load(file)

    for k, v in data.items():
        cache_results(k, v)

print("Successfully restored Redis data from redis_dump.json")
