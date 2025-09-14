import json
import os


def classify_city(city_name: str) -> int:
    filepath = os.path.join(os.path.dirname(__file__), "cities.json")
    with open(filepath, "r", encoding="utf-8") as f:
        cities = json.load(f)
    megapolises = {"Астана", "Алматы", "Шымкент"}
    if city_name in megapolises:
        return 1
    elif city_name in cities:
        return 2 if cities[city_name] >= 500_000 else 3
    else:
        return 3 