import json

import redis

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

keys = redis_client.keys("*")

data = {}

for key in keys:
    data[key] = redis_client.get(key)

with open("redis_dump.json", "w") as json_file:
    json.dump(data, json_file, indent=2)

print("Redis data successfully saved to redis_dump.json")
