# -*- coding: utf-8 -*-
# recommender.py
"""
Как запускать (примеры):

1) Один клиент (JSON c фичами):
python recommender.py --models ./models --model-file ./models/lgbm_all.pkl --features ./samples/client_123.json --topk 3 --interactive

2) Пакет клиентов (CSV):
python recommender.py --models ./models --model-file ./models/lgbm_all.pkl --batch ./samples/clients.csv --csv-out ./out/recommendations.csv --topk 3

Ожидаемый INPUT (минимум для корректной персонализации):
JSON (пример ./samples/client_123.json):
{
  "client_id": 123,
  "name": "Рамазан",
  "month_num": 8,
  "status": "Стандартный клиент",
  "city": "Алматы",
  "avg_monthly_balance_KZT": 2000000,
  "spend_by_category": { "Такси": 27400, "Путешествия": 51000, "Рестораны": 65000, "Едим дома": 12000, "Смотрим дома": 8000 },
  "spend_total": 350000,
  "fx_preferred": "EUR",               # или null
  "trips_taxi_count": 12,
  "online_home_spend": 32000,          # Едим/Смотрим/Играем дома — суммарно
  "top_categories": ["Рестораны","Такси","Продукты питания"],
  "atm_withdrawals_cnt": 3,
  "transfers_cnt": 5,
  "cash_gap_kzt": 0,                   # >0 если кассовый разрыв
  "free_money_kzt": 50000,             # свободный остаток/накопления
  "fx_activity": 1,                    # 1 если есть fx_buy/fx_sell
  "installment_or_cc_flags": 0         # 1 если installment/cc_repayment встречались
}

CSV для батча — те же поля колонками (минимум: client_id,name,month_num,avg_monthly_balance_KZT,spend_total,top_categories как ;-разделённый список).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import csv
from typing import Dict, Any, List, Tuple, Optional

# МЛ: поддерживаем joblib/pickle и LightGBM Booster
try:
    import joblib
except Exception:
    joblib = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# наш генератор пушей из предыдущего шага (лежит рядом)
try:
    from push_generator import generate_push
except ImportError:
    print("❌ Не найден push_generator.py рядом со скриптом. Положи файл рядом и повтори запуск.")
    sys.exit(1)

# ===== Константы продуктов по ТЗ =================================================

PRODUCTS = [
    "Карта для путешествий",
    "Премиальная карта",
    "Кредитная карта",
    "Обмен валют",
    "Депозит мультивалютный",
    "Депозит сберегательный",
    "Депозит накопительный",
    "Инвестиции",
    "Кредит наличными",
    "Золотые слитки",
]

MEGAPOLISES_KZ = {"Астана", "Алматы", "Шымкент"}

# ===== Вспомогательные утилиты ===================================================

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        # попытка расклеить строковый список
        return [t.strip() for t in x.split(";") if t.strip()]
    return []

def spend(spend_by_category: Dict[str, Any], cats: List[str]) -> float:
    return sum(_safe_float(spend_by_category.get(c, 0)) for c in cats)

def entropy_from_shares(shares: List[float]) -> float:
    # простая энтропия Шеннона для оценки разнообразия трат
    import math
    s = [x for x in shares if x > 0]
    if not s:
        return 0.0
    return -sum(p * math.log(p + 1e-12) for p in s)

# ===== Eligibility и бизнес-правила по ТЗ =======================================

def eligibility(product: str, f: Dict[str, Any]) -> bool:
    """Жёсткие фильтры, когда продукт показывать не стоит."""
    status = f.get("status", "")
    avg_bal = _safe_float(f.get("avg_monthly_balance_KZT", 0))
    spend_by_category = f.get("spend_by_category", {}) or {}
    total_spend = _safe_float(f.get("spend_total", 0))
    cash_gap = _safe_float(f.get("cash_gap_kzt", 0))
    fx_pref = f.get("fx_preferred")
    fx_act = int(f.get("fx_activity", 0))
    online_home = _safe_float(f.get("online_home_spend", 0))
    install_cc = int(f.get("installment_or_cc_flags", 0))

    if product == "Кредит наличными":
        # только при явной потребности
        return cash_gap > 0

    if product == "Премиальная карта":
        # условно «высокий остаток»
        return avg_bal >= 1_000_000

    if product == "Обмен валют":
        # есть интерес к валюте: предпочитаемая валюта или активность
        return bool(fx_pref) or fx_act == 1

    if product == "Кредитная карта":
        # осмысленно, если есть ярко выраженные топ-категории или онлайн-сервисы/рассрочка
        top_cats = _as_list(f.get("top_categories"))
        strong_prefs = len(top_cats) >= 2 or online_home > 0 or install_cc == 1
        return strong_prefs and total_spend > 0

    if product == "Карта для путешествий":
        # есть траты на Такси/Путешествия/Отели
        s = spend(spend_by_category, ["Такси", "Путешествия", "Отели"])
        return s > 0

    if product in ("Депозит мультивалютный", "Депозит сберегательный", "Депозит накопительный", "Инвестиции"):
        # осмысленно при наличии свободных средств
        free_money = _safe_float(f.get("free_money_kzt", 0))
        return free_money > 0 or avg_bal > 0

    # по остальным — показываем по умолчанию
    return True

def business_score_boost(product: str, f: Dict[str, Any]) -> float:
    """Мягкие надбавки/штрафы к ML-скорингу по сигналам из ТЗ."""
    spend_by_category = f.get("spend_by_category", {}) or {}
    avg_bal = _safe_float(f.get("avg_monthly_balance_KZT", 0))
    online_home = _safe_float(f.get("online_home_spend", 0))
    top_cats = _as_list(f.get("top_categories"))
    fx_pref = f.get("fx_preferred")
    fx_act = int(f.get("fx_activity", 0))
    s_travel = spend(spend_by_category, ["Такси", "Путешествия", "Отели"])
    s_premium = spend(spend_by_category, ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"])

    boost = 0.0
    if product == "Карта для путешествий":
        boost += 0.001 * s_travel  # чем больше трат на дорогу — тем выше скор

    if product == "Премиальная карта":
        boost += 0.0005 * avg_bal + 0.001 * s_premium

    if product == "Кредитная карта":
        boost += 0.001 * online_home
        boost += 0.0005 * sum(len(c) for c in top_cats[:3])  # «выраженность» предпочтений

    if product == "Обмен валют":
        if fx_pref: boost += 0.5
        if fx_act == 1: boost += 0.5

    if product in ("Депозит мультивалютный", "Депозит сберегательный", "Депозит накопительный", "Инвестиции"):
        free_money = _safe_float(f.get("free_money_kzt", 0))
        boost += 0.0008 * max(free_money, 0)

    if product == "Кредит наличными":
        cash_gap = _safe_float(f.get("cash_gap_kzt", 0))
        boost += 0.001 * max(cash_gap, 0)

    return boost

# ===== Загрузка моделей ==========================================================

def load_models(models_dir: str, model_file: Optional[str]) -> Tuple[str, Any]:
    """
    Возвращает ('single', model) или ('per_product', dict[product]->model)
    Поддержка:
      - единый файл: .pkl/.joblib (sklearn/lightgbm wrapper), .txt/.json (LightGBM Booster)
      - папка с файлами: <models_dir>/<product>.pkl|.joblib|.txt|.json
    """
    def _load_one(path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".pkl", ".joblib"):
            if not joblib:
                raise RuntimeError("joblib не установлен, не могу загрузить pkl/joblib")
            return joblib.load(path)
        if ext in (".txt", ".json"):
            if not lgb:
                raise RuntimeError("lightgbm не установлен, не могу загрузить Booster")
            booster = lgb.Booster(model_file=path)
            return booster
        raise RuntimeError(f"Неподдерживаемое расширение модели: {ext}")

    # Случай 1: явно указан единый файл
    if model_file:
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Модель не найдена: {model_file}")
        return "single", _load_one(model_file)

    # Случай 2: ищем по продуктам в папке
    models = {}
    for p in PRODUCTS:
        basename = p
        for ext in (".pkl", ".joblib", ".txt", ".json"):
            candidate = os.path.join(models_dir, basename + ext)
            if os.path.isfile(candidate):
                models[p] = _load_one(candidate)
                break
    if models:
        return "per_product", models

    raise RuntimeError("Не нашёл ни единой модели. Укажи --model-file или положи модели per product в --models.")

# ===== Преобразование фич в вектор для модели ===================================

def vectorize_features(f: Dict[str, Any], feature_order: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Приведите к тому же порядку/набору фич, что использовались при обучении (feature_order).
    Если order не задан — формируем минимальный, типовой набор (можно расширять под свой пайплайн).
    """
    sbc = f.get("spend_by_category", {}) or {}
    feats = {
        "avg_balance": _safe_float(f.get("avg_monthly_balance_KZT", 0)),
        "spend_total": _safe_float(f.get("spend_total", 0)),
        "spend_travel": spend(sbc, ["Путешествия", "Отели", "Такси"]),
        "spend_premium": spend(sbc, ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"]),
        "spend_online_home": _safe_float(f.get("online_home_spend", 0)),
        "atm_withdrawals_cnt": _safe_float(f.get("atm_withdrawals_cnt", 0)),
        "transfers_cnt": _safe_float(f.get("transfers_cnt", 0)),
        "fx_activity": _safe_float(f.get("fx_activity", 0)),
        "cash_gap_kzt": _safe_float(f.get("cash_gap_kzt", 0)),
        "free_money_kzt": _safe_float(f.get("free_money_kzt", 0)),
        "installment_or_cc_flags": _safe_float(f.get("installment_or_cc_flags", 0)),
    }
    # Категориальные можно было кодировать при подготовке датасета; здесь предполагаем, что модель ждёт численные фичи
    if feature_order:
        # отфильтровать/упорядочить
        feats = {k: feats.get(k, 0.0) for k in feature_order}
    return feats

# ===== Предсказание benefit-скорингов моделью ===================================

def predict_product_score(model_obj, vec: Dict[str, float]) -> float:
    """Универсальный предикт: поддержка Booster и sklearn-like моделей."""
    import numpy as np
    X = np.array([list(vec.values())], dtype=float)
    # LightGBM Booster
    if lgb and isinstance(model_obj, lgb.Booster):
        y = model_obj.predict(X)
        return float(y[0] if hasattr(y, "__len__") else y)
    # sklearn / LGBMRegressor/Classifier via joblib
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(X)
        # берём вероятность класса 1, если бинарка
        if isinstance(proba, list):  # на всякий
            proba = proba[0]
        if hasattr(proba, "__len__") and len(proba[0]) > 1:
            return float(proba[0][1])
        return float(proba[0] if hasattr(proba[0], "__len__") else proba)
    if hasattr(model_obj, "predict"):
        y = model_obj.predict(X)
        return float(y[0] if hasattr(y, "__len__") else y)
    raise RuntimeError("Не понимаю тип модели: нет predict/predict_proba/Booster")

def predict_all_products(model_mode: str, model_obj, feats: Dict[str, float]) -> Dict[str, float]:
    """Возвращает словарь product -> ml_score."""
    if model_mode == "single":
        # единая модель: считаем, что она предсказывает «общую полезность»; разнесём по продуктам одинаково
        base = predict_product_score(model_obj, feats)
        return {p: base for p in PRODUCTS}
    elif model_mode == "per_product":
        scores = {}
        for p, m in model_obj.items():
            scores[p] = predict_product_score(m, feats)
        # если не все продукты покрыты моделями — добьём нулями
        for p in PRODUCTS:
            scores.setdefault(p, 0.0)
        return scores
    else:
        raise RuntimeError("Неизвестный режим модели")

# ===== Ранжирование с бизнес-правилами ==========================================

def rank_products(raw_scores: Dict[str, float], f: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    """
    Возвращает список (product, ml_score, boost, final_score) отсортированный по final_score desc,
    с учётом eligibility.
    """
    ranked = []
    for p, s in raw_scores.items():
        if not eligibility(p, f):
            continue
        b = business_score_boost(p, f)
        final = s + b
        ranked.append((p, float(s), float(b), float(final)))
    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked

# ===== Генерация push текста под top-1 ==========================================

def build_push_for_top(product: str, f: Dict[str, Any]) -> str:
    name = f.get("name", "Клиент")
    month_num = int(f.get("month_num", 0) or 0) or 8  # дефолт август, если не передали

    # Достаём необходимые поля для разных продуктов — делаем лучшее, что можем из доступных фич
    spend_by_category = f.get("spend_by_category", {}) or {}
    taxi_cnt = int(f.get("trips_taxi_count", 0))
    s_taxi = int(spend(spend_by_category, ["Такси"]))
    s_travel = int(spend(spend_by_category, ["Путешествия","Отели"]) + s_taxi)
    cashback_est = max(int(0.04 * s_travel), 0)

    if product == "Карта для путешествий":
        return generate_push(product,
            name=name, month_num=month_num,
            trips_count=taxi_cnt, spend_kzt=max(s_travel, s_taxi), cashback_kzt=max(cashback_est, 0))

    if product == "Премиальная карта":
        return generate_push(product,
            name=name,
            avg_balance_kzt=int(_safe_float(f.get("avg_monthly_balance_KZT", 0))),
            highlight_spend="рестораны" if spend_by_category.get("Кафе и рестораны") else "повседневные покупки"
        )

    if product == "Кредитная карта":
        top_cats = _as_list(f.get("top_categories"))
        if not top_cats:
            # подхватим 2–3 категории с наибольшими spend
            top_cats = sorted(spend_by_category.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_cats = [k for k, _ in top_cats]
        return generate_push(product, name=name, top_categories=top_cats)

    if product == "Обмен валют":
        curr = f.get("fx_preferred") or "USD"
        return generate_push(product, name=name, currency=curr)

    if product == "Депозит мультивалютный":
        return generate_push(product, name=name)

    if product == "Депозит сберегательный":
        return generate_push(product, name=name, free_kzt=int(_safe_float(f.get("free_money_kzt", 0))))

    if product == "Депозит накопительный":
        monthly_topup = int(_safe_float(f.get("free_money_kzt", 0)))
        return generate_push(product, name=name, monthly_topup_kzt=monthly_topup)

    if product == "Инвестиции":
        start = int(_safe_float(f.get("free_money_kzt", 0)) or _safe_float(f.get("avg_monthly_balance_KZT", 0)) * 0.1)
        return generate_push(product, name=name, start_kzt=max(start, 10000))

    if product == "Кредит наличными":
        gap = int(_safe_float(f.get("cash_gap_kzt", 0)))
        return generate_push(product, name=name, cash_gap_kzt=max(gap, 0))

    if product == "Золотые слитки":
        return generate_push(product, name=name)

    # fallback (не должен сработать)
    return f"{name}, для вас подобран продукт: {product}. Посмотреть условия."

# ===== IO: загрузка одного клиента / батча ======================================

def load_features_from_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # нормализуем spend_by_category если строка
    if isinstance(data.get("spend_by_category"), str):
        try:
            data["spend_by_category"] = json.loads(data["spend_by_category"])
        except Exception:
            data["spend_by_category"] = {}
    return data

def iter_clients_from_csv(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # приведём некоторые поля
            for k in ("avg_monthly_balance_KZT","spend_total","online_home_spend",
                      "atm_withdrawals_cnt","transfers_cnt","cash_gap_kzt","free_money_kzt"):
                if k in row:
                    row[k] = _safe_float(row[k], 0)
            if "top_categories" in row:
                row["top_categories"] = _as_list(row["top_categories"])
            if "spend_by_category" in row and isinstance(row["spend_by_category"], str):
                try:
                    row["spend_by_category"] = json.loads(row["spend_by_category"])
                except Exception:
                    row["spend_by_category"] = {}
            yield row

# ===== Основной сценарий =========================================================

def recommend_for_client(models_dir: str, model_file: Optional[str], features: Dict[str, Any], topk: int = 1,
                         feature_order: Optional[List[str]] = None) -> Dict[str, Any]:
    model_mode, model_obj = load_models(models_dir, model_file)
    vec = vectorize_features(features, feature_order)
    ml_scores = predict_all_products(model_mode, model_obj, vec)
    ranked = rank_products(ml_scores, features)

    result = {
        "client_id": features.get("client_id"),
        "name": features.get("name"),
        "recommendations": []
    }

    for i, (product, ml_s, boost, final) in enumerate(ranked[:max(1, topk)], start=1):
        push_text = build_push_for_top(product, features)
        result["recommendations"].append({
            "rank": i,
            "product": product,
            "ml_score": round(ml_s, 6),
            "business_boost": round(boost, 6),
            "final_score": round(final, 6),
            "push_notification": push_text
        })
    return result

def main():
    ap = argparse.ArgumentParser(description="Рекомендации продуктов + генерация пушей (по ТЗ).")
    ap.add_argument("--models", required=True, help="Папка с моделями (per product) или просто существующая папка.")
    ap.add_argument("--model-file", default=None, help="Единый файл модели (joblib/pkl или lgbm txt/json).")
    ap.add_argument("--features", help="JSON с фичами одного клиента.")
    ap.add_argument("--batch", help="CSV с фичами множества клиентов.")
    ap.add_argument("--csv-out", help="Путь для сохранения CSV с рекомендациями (для --batch).")
    ap.add_argument("--topk", type=int, default=1, help="Сколько рекомендаций возвращать.")
    ap.add_argument("--interactive", action="store_true", help="Печать понятного вывода в консоль.")
    args = ap.parse_args()

    if not args.features and not args.batch:
        print("Нужно указать либо --features <file.json>, либо --batch <file.csv>")
        sys.exit(2)

    if args.features:
        feats = load_features_from_json(args.features)
        rec = recommend_for_client(args.models, args.model_file, feats, topk=args.topk)

        if args.interactive:
            print("—" * 70)
            print(f"Клиент: {rec.get('name')} (id={rec.get('client_id')})")
            for r in rec["recommendations"]:
                print(f"\n#{r['rank']} → {r['product']}  | final={r['final_score']}  (ml={r['ml_score']} / boost={r['business_boost']})")
                print(r["push_notification"])
            print("—" * 70)
        else:
            print(json.dumps(rec, ensure_ascii=False, indent=2))

    if args.batch:
        if not args.csv_out:
            print("Для батча укажи --csv-out <path.csv>, чтобы сохранить результаты.")
            sys.exit(2)

        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", encoding="utf-8", newline="") as fw:
            writer = csv.writer(fw)
            writer.writerow(["client_id", "name", "rank", "product", "ml_score", "business_boost", "final_score", "push_notification"])
            for feats in iter_clients_from_csv(args.batch):
                rec = recommend_for_client(args.models, args.model_file, feats, topk=args.topk)
                for r in rec["recommendations"]:
                    writer.writerow([
                        rec.get("client_id"), rec.get("name"), r["rank"], r["product"],
                        r["ml_score"], r["business_boost"], r["final_score"], r["push_notification"]
                    ])
        print(f"✅ Результаты сохранены: {args.csv_out}")

if __name__ == "__main__":
    main()
