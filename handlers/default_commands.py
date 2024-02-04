from aiogram import Router, types
from aiogram import html
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from keyboards.start_keyboard import get_start_keyboard
from handlers.dialog import start_state

router = Router()

# Чтобы функция стала обработчиком события, нужно
# оформить ее через специальный декоратор
# или зарегистрировать ее у диспетчера или роутера


# Обработчик события (хэндлер) на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    # Определяем имя пользователя и выводим приветствие
    await message.answer(
        f"Привет, {html.bold(html.quote(message.from_user.full_name))}!",
        parse_mode=ParseMode.HTML,
        reply_markup=get_start_keyboard(),
    )
    # и переходим в начальное состояние
    await start_state(message, state)


# Хэндлер на команду /help
@router.message(Command("help"))
async def cmd_help(message: types.Message, state: FSMContext):
    await message.answer(
        "Бот предназначен для демонстрации применения нейронных сетей для стилизации изображений.\n"
        "\n"
        "Предусмотрены следующие варианты стилизации:\n"
        "1. Алгоритм постепенного переноса стиля с примением нейронной сети VGG16 и матрицы Грама.\n"
        "В этом алгоритме на вход подается изображение (контент), которое будет преобразовано "
        "в соответствии со стилем второго изображения. Работает медленно.\n"
        "2. Преобразование к стилю художника Ван Гога. Используется предобученная в UC Berkeley "
        "нейронная сеть по методу cycle GAN.\n"
        "3. Преобразование зимнего пейзажа к летнему. Также используется сеть CGAN, предобученная в UC Berkeley.\n"
        "\n"
        "После нажатия кнопки 'Начать заново', "
        "предлагается выбрать способ преобразования "
        "и загрузить одно или два изображения.\n"
        "Изображения необходимо передавать как фотографии, а не как файлы.\n"
        "Для ускорения работы, изображения уменьшаются по размеру.\n"
        "\n"
        "Для начала работы, нажмите кнопку 'Начать заново' на выпадающей клавиатуре снизу..."
    )


# Обработчик события (хэндлер) на команду /transfer_style
@router.message(Command("transfer_style"))
async def transfer_style(message: types.Message, state: FSMContext):
    # переход в начальное состояние
    await start_state(message, state)
