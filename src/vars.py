CATEGORIES: dict = {
    'Одежда и обувь': 'fashion',
    'Продукты питания': 'food',
    'Кафе и рестораны': 'food',
    'Медицина': 'health',
    'Авто': 'vehicle',
    'Спорт': 'sports',
    'Развлечения': 'entertainment',
    'АЗС': 'fuel',
    'Кино': 'cinema',
    'Питомцы': 'pets',
    'Книги': 'books',
    'Цветы': 'flowers',
    'Едим дома': 'onlineservices',
    'Смотрим дома': 'onlineservices',
    'Играем дома': 'onlineservices',
    'Косметика и Парфюмерия': 'premiumgoods',
    'Подарки': 'gifts',
    'Ремонт дома': 'homerepair',
    'Мебель': 'furniture',
    'Спа и массаж': 'wellness',
    'Ювелирные украшения': 'premiumgoods',
    'Такси': 'transport',
    'Отели': 'travel',
    'Путешествия': 'travel'
}

ALL_CATS: list = list(set(CATEGORIES.values()))
ALL_TRANSFS = [
    'salary_in','stipend_in','family_in','cashback_in','refund_in','card_in',
    'p2p_out','card_out','atm_withdrawal','utilities_out','loan_payment_out',
    'cc_repayment_out','installment_payment_out','fx_buy','fx_sell',
    'invest_out','invest_in','deposit_topup_out','deposit_fx_topup_out',
    'deposit_fx_withdraw_in','gold_buy_out','gold_sell_in'
]