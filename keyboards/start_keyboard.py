from aiogram import types


# Определим структуру главного выпадающего меню
def get_start_keyboard() -> types.ReplyKeyboardMarkup:

    kb = [
        [
            types.KeyboardButton(text="Начать заново"),
            types.KeyboardButton(text="Помощь")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите действие"
    )

    return keyboard

