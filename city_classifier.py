import json, argparse, csv, sys
from pathlib import Path

def classify(pop, mega=1_000_000, large=250_000):
    if pop is None: return 3
    return 1 if pop > mega else (2 if pop >= large else 3)

def load_data(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() in [".csv", ".tsv"]:
        delim = "," if p.suffix.lower()==".csv" else "\t"
        out = {}
        with p.open(encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            # ожидаем колонки: city, population
            for row in r:
                name = row.get("city") or row.get("город") or row.get("name")
                pop_raw = row.get("population") or row.get("население")
                pop = int(pop_raw.replace(" ", "")) if pop_raw and pop_raw.strip() else None
                if name: out[name] = pop
        return out
    else:
        print("Поддерживаются .json / .csv / .tsv", file=sys.stderr)
        sys.exit(2)

def main():
    ap = argparse.ArgumentParser(description="Классифицирует города РК → {1,2,3}")
    ap.add_argument("--input", required=True, help="Путь к cities.json|csv (город → население)")
    ap.add_argument("--output", default="city_classes.json", help="Куда сохранить результат")
    ap.add_argument("--mega", type=int, default=1_000_000, help="Порог мегаполиса (default: 1_000_000)")
    ap.add_argument("--large", type=int, default=250_000, help="Порог крупного города (default: 250_000)")
    args = ap.parse_args()

    data = load_data(args.input)
    result = {city: classify(pop, args.mega, args.large) for city, pop in data.items()}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: записал {len(result)} записей в {args.output}")

if __name__ == "__main__":
    main()
