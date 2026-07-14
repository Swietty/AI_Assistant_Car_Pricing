import datetime


# =====================================
# Класс автомобиля
# =====================================

brand_class = {


    # Luxury
    "bentley": 5,
    "ferrari": 5,
    "lamborghini": 5,
    "aston-martin": 5,


    # Premium
    "porsche": 4,
    "maserati": 4,
    "mercedes-benz": 4,
    "bmw": 4,
    "audi": 4,
    "land-rover": 4,
    "jaguar": 4,
    "cadillac": 4,


    # Upper middle
    "alfa-romeo": 3,
    "mini": 3,
    "volvo": 3,
    "infiniti": 3,


    # Middle
    "volkswagen": 2,
    "skoda": 2,
    "seat": 2,
    "toyota": 2,
    "mazda": 2,
    "honda": 2,
    "ford": 2,
    "nissan": 2,
    "hyundai": 2,
    "kia": 2,
    "opel": 2,
    "renault": 2,
    "peugeot": 2,
    "citroen": 2,


    # Budget
    "dacia": 1,
    "lada": 1,
    "daewoo": 1,
    "daihatsu": 1,
    "proton": 1,
    "ssangyong": 1

}



# =====================================
# Базовое влияние повреждения %
# =====================================

damage_base = {


    "Lamp broken": 5,


    "Glass shatter": 4,


    "Dent": 3,


    "Scratch": 1.5,


    "Flat tire": 1,


    "Crack": 2

}



# =====================================
# Цена ремонта относительно класса
# =====================================

brand_factor = {


    1: 0.7,    # Budget

    2: 1.0,    # Normal

    3: 1.2,    # Upper middle

    4: 1.5,    # Premium

    5: 2.0     # Luxury

}



# =====================================
# Возраст
# =====================================

def age_factor(year):


    current_year = datetime.datetime.now().year

    age = current_year - year



    if age <= 2:

        return 1.5


    elif age <= 5:

        return 1.3


    elif age <= 10:

        return 1.0


    else:

        return 0.7




# =====================================
# Пробег
# =====================================

def mileage_factor(mileage):


    if mileage < 50000:

        return 1.3


    elif mileage < 150000:

        return 1.0


    else:

        return 0.8




# =====================================
# Главная функция
# =====================================

def calculate_damage_discount(

        brand,
        year,
        mileage,
        damage_type,
        confidence

):


    if damage_type not in damage_base:

        return 0



    base_damage = damage_base[damage_type]



    car_class = brand_class.get(

        brand.lower(),

        2

    )



    discount = (

        base_damage

        *

        brand_factor[car_class]

        *

        age_factor(year)

        *

        mileage_factor(mileage)

        *

        confidence

    )



    # максимум потери

    if discount > 20:

        discount = 20



    return round(
        discount,
        1
    )