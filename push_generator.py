# -*- coding: utf-8 -*-
# push_generator.py
from __future__ import annotations
import random
from typing import List, Optional

# ==== ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ =================================================

_RU_MONTHS_GEN = {
    1: "январе", 2: "феврале", 3: "марте", 4: "апреле",
    5: "мае", 6: "июне", 7: "июле", 8: "августе",
    9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре",
}

def month_genitive(month_num: int) -> str:
    """Возвращает название месяца в родительном падеже (для 'в августе')."""
    return _RU_MONTHS_GEN.get(month_num, "текущем месяце")

def _fmt_int_with_spaces(n: int) -> str:
    # 27400 -> "27 400"
    return f"{n:,}".replace(",", " ")

def fmt_kzt(amount: float, decimals: int = 0, approx: bool = False) -> str:
    """
    Формат валюты по редполитике:
    - пробелы-разряды
    - запятая как дробная часть
    - знак валюты отделён пробелом: 2 490 ₸
    """
    if decimals == 0:
        s = _fmt_int_with_spaces(int(round(amount)))
    else:
        s = f"{amount:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"≈{s} ₸" if approx else f"{s} ₸"

def pluralize(n: int, forms: List[str]) -> str:
    """
    Русская множественность: forms = ["поездка","поездки","поездок"]
    """
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 11 <= n_abs <= 14: return forms[2]
    if n1 == 1: return forms[0]
    if 2 <= n1 <= 4: return forms[1]
    return forms[2]

# ==== ШАБЛОНЫ ПО ПРОДУКТАМ ====================================================

def push_travel_card(
    name: str,
    month_num: int,
    trips_count: int,
    spend_kzt: int,
    cashback_kzt: int
) -> str:
    month = month_genitive(month_num)
    trips_word = pluralize(trips_count, ["поездку", "поездки", "поездок"])
    spend = fmt_kzt(spend_kzt)
    cashback = fmt_kzt(cashback_kzt, approx=True)

    templates = [
        f"{name}, в {month} у вас {trips_count} {trips_word} на такси на {spend}. "
        f"С картой для путешествий вернулось бы {cashback}. Откройте карту в приложении.",

        f"{name}, много поездок в {month}. С тревел-картой часть расходов на дорогу вернулась бы кешбэком {cashback}. "
        f"Оформите карту.",

        f"{name}, дорога в {month} обошлась в {spend}. Тревел-карта сэкономила бы вам {cashback}. "
        f"Подключите сейчас."
    ]
    return random.choice(templates)

def push_premium_card(
    name: str,
    avg_balance_kzt: int,
    highlight_spend: str
) -> str:
    bal = fmt_kzt(avg_balance_kzt)
    templates = [
        f"{name}, у вас стабильно крупный остаток ({bal}) и траты на {highlight_spend}. "
        f"Премиальная карта даст до 4% кешбэка и бесплатные снятия. Подключите сейчас.",

        f"{name}, при остатке около {bal} премиальная карта увеличит кешбэк и снимет комиссии за снятия/переводы. "
        f"Оформите карту."
    ]
    return random.choice(templates)

def push_credit_card(
    name: str,
    top_categories: List[str]
) -> str:
    cats = ", ".join(top_categories[:3])
    templates = [
        f"{name}, ваши топ-траты — {cats}. Кредитная карта вернула бы до 10% именно там и на онлайн-сервисы. "
        f"Оформите карту.",

        f"{name}, чаще всего вы платите за {cats}. Кредитная карта вернёт заметный кешбэк и даст льготный период. "
        f"Подключите сейчас."
    ]
    return random.choice(templates)

def push_fx(
    name: str,
    currency: str
) -> str:
    # currency, например: "USD" или "EUR"
    return (
        f"{name}, вы часто платите в {currency}. В приложении — выгодный обмен и авто-покупка по вашему курсу. "
        f"Настроить обмен."
    )

def push_deposit_saving(
    name: str,
    free_kzt: int
) -> str:
    free_s = fmt_kzt(free_kzt, approx=True)
    return (
        f"{name}, у вас остаются свободные средства ({free_s}). "
        f"Сберегательный вклад даёт повышенную ставку за счёт «заморозки». Открыть вклад."
    )

def push_deposit_accum(
    name: str,
    monthly_topup_kzt: int
) -> str:
    topup = fmt_kzt(monthly_topup_kzt, approx=True)
    return (
        f"{name}, удобно копить по чуть-чуть: пополняйте на {topup} — деньги работают, снятие не предусмотрено. "
        f"Открыть накопительный вклад."
    )

def push_investments(
    name: str,
    start_kzt: int
) -> str:
    start = fmt_kzt(start_kzt)
    return (
        f"{name}, попробуйте инвестиции с низким порогом входа — начните с {start} и без комиссий на старт. "
        f"Открыть счёт."
    )

def push_cash_loan(
    name: str,
    cash_gap_kzt: int
) -> str:
    gap = fmt_kzt(cash_gap_kzt)
    return (
        f"{name}, в этом месяце расходы превысили поступления на {gap}. "
        f"Кредит наличными поможет закрыть разрыв с гибкими выплатами. Узнайте лимит."
    )

def push_gold(
    name: str
) -> str:
    return (
        f"{name}, золотые слитки — простой способ защитить сбережения и диверсифицировать портфель. "
        f"Посмотреть условия."
    )

def push_multicurrency_deposit(
    name: str
) -> str:
    return (
        f"{name}, мультивалютный вклад — проценты по счёту и удобно держать валюту под задачи. "
        f"Открыть вклад."
    )

# ==== ЕДИНАЯ ТОЧКА ВХОДА ======================================================

def generate_push(product: str, **kwargs) -> str:
    """
    Универсальный генератор:
      product: одно из
        - "Карта для путешествий"
        - "Премиальная карта"
        - "Кредитная карта"
        - "Обмен валют"
        - "Депозит сберегательный"
        - "Депозит накопительный"
        - "Инвестиции"
        - "Кредит наличными"
        - "Золотые слитки"
        - "Депозит мультивалютный"
      kwargs: параметры для соответствующей функции (см. сигнатуры выше)
    """
    mapping = {
        "Карта для путешествий": push_travel_card,
        "Премиальная карта": push_premium_card,
        "Кредитная карта": push_credit_card,
        "Обмен валют": push_fx,
        "Депозит сберегательный": push_deposit_saving,
        "Депозит накопительный": push_deposit_accum,
        "Инвестиции": push_investments,
        "Кредит наличными": push_cash_loan,
        "Золотые слитки": push_gold,
        "Депозит мультивалютный": push_multicurrency_deposit,
    }
    fn = mapping.get(product)
    if not fn:
        raise ValueError(f"Неизвестный продукт: {product}")
    return fn(**kwargs)


# ==== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================================================
if __name__ == "__main__":
    # travel
    print(generate_push(
        "Карта для путешествий",
        name="Рамазан", month_num=8, trips_count=12, spend_kzt=27400, cashback_kzt=1100
    ))
    # premium
    print(generate_push(
        "Премиальная карта",
        name="Алия", avg_balance_kzt=2_000_000, highlight_spend="рестораны"
    ))
    # credit
    print(generate_push(
        "Кредитная карта",
        name="Данияр", top_categories=["продукты", "такси", "онлайн-сервисы"]
    ))
    # fx
    print(generate_push(
        "Обмен валют",
        name="Асет", currency="EUR"
    ))
