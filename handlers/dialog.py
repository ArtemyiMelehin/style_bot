import os
from aiogram import Bot, types, F, Router, html
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from tempfile import gettempdir

from handlers import default_commands
from transform_models.style_transfer_gram import Model_style_transfer_gram
from transform_models.cycleGAN import Model_style_transfer_CGAN

# определим пусть к директории для временных файлов
temp_dir = gettempdir()  # '/tmp/'

router = Router()

# Определим класс состояний пользователя для отслеживания в диалоге
class UserStates(StatesGroup):
    start = State()
    send_content_img = State()
    send_style_img = State()
    generate_img = State()


# Идентификаторы и описание стилей
style_descr = {
    'vangogh': 'Стиль художника Ван Гога',
    'winter2summer': 'Преобразование зимы в лето',
}


# Функция для создания инлайн-клавиатуры
def select_keyboard() -> types.InlineKeyboardMarkup:
    buttons = [
        [types.InlineKeyboardButton(text='"Медленный" перенос стиля', 
                                    callback_data="style_gram")],
        [types.InlineKeyboardButton(text=style_descr['vangogh'], 
                                    callback_data="cgan_style_vangogh")],
        [types.InlineKeyboardButton(text=style_descr['winter2summer'], 
                                    callback_data="cgan_style_winter2summer")],
    ]
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard


@router.message(F.text.lower() == "начать заново")
async def button_restart(message: types.Message, state: FSMContext) -> None:
    await default_commands.cmd_start(message, state)


@router.message(F.text.lower() == "помощь")
async def button_help(message: types.Message, state: FSMContext) -> None:
    await default_commands.cmd_help(message, state)


# Обработчик начального состояния
@router.message(UserStates.start)
async def start_state(message: types.Message, state: FSMContext) -> None:
    await state.set_state(UserStates.send_content_img)
    # выводим сообщение и прикрепляем инлайн-клавиатуру к нему
    await message.answer(
        text="Выберите вариант стилизации:", reply_markup=select_keyboard()
    )

# Если нажата кнопка style_gram
@router.callback_query(F.data == "style_gram")
async def select_style_gram(callback: types.CallbackQuery, state: FSMContext) -> None:
    await state.update_data(model="gram")
    # устанавливаем новое состояние
    await state.set_state(UserStates.send_content_img)
    # изменим текст в сообщении где была нажата кнопка
    await callback.message.edit_text(
        text='Выбран "медленный" перенос стиля.\n'
        "Пришлите изображение, которое будем стилизовать..."
    )
    await callback.answer()


# Если нажата одна из кнопок, где data начинается на "cgan_style_"
@router.callback_query(F.data.startswith("cgan_style_"))
async def cgan_style(callback: types.CallbackQuery, state: FSMContext) -> None:
    # определяем имя стиля, разделив по знаку _
    style_name = callback.data.split("_")[2]
    # обновляем переменные model и cgan_style_name
    await state.update_data(model="CGAN")
    await state.update_data(cgan_style_name=style_name)
    # и меняем состояние
    await state.set_state(UserStates.send_content_img)
    # берем описание стиля
    descr_text = style_descr[style_name]
    # и изменим текст в сообщении где была нажата кнопка
    await callback.message.edit_text(
        f"Выбрано: {html.bold(html.quote(descr_text))}.\n"
        "Пришлите изображение, которое будем стилизовать...",
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


# Если состояние send_content_img и пользователь прислал изображение
@router.message(UserStates.send_content_img, F.photo)
async def download_content_img(
    message: types.Message, bot: Bot, state: FSMContext
) -> None:
    user_data = await state.get_data()
    # изображения приходят сразу в разных разрешениях
    hi_res_foto = message.photo[-1]  # последнее в списке в высоком разрешении
    img_file_name = os.path.join(temp_dir, f"{hi_res_foto.file_id}_cont.jpg")
    await bot.download(hi_res_foto, destination=img_file_name)
    await state.update_data(content_img_file_name=img_file_name)

    # Если выбрана модель CGAN, то сразу начинаем генерацию
    if user_data["model"] == "CGAN":
        await message.answer("Изображение (content) принято.")
        await state.set_state(UserStates.generate_img)
        await generate_img(message, state)
    # иначе, обновим состояние и попросим прислать изображение стиля
    else:
        await message.answer(
            "Изображение (content) принято. Теперь пришлите изображение стиля..."
        )
        await state.set_state(UserStates.send_style_img)


# Обрабатываем изображение стиля, при условии если состояние = send_style_img
@router.message(UserStates.send_style_img, F.photo)
async def download_style_img(
    message: types.Message, bot: Bot, state: FSMContext
) -> None:
    hi_res_foto = message.photo[-1]  # последнее в списке в высоком разрешении
    img_file_name = os.path.join(temp_dir, f"{hi_res_foto.file_id}_style.jpg")
    await bot.download(hi_res_foto, destination=img_file_name)
    await state.update_data(style_img_file_name=img_file_name)
    await message.answer("Изображение (стиль) принято.")
    await state.set_state(UserStates.generate_img)
    await generate_img(message, state)


# Запуск генерации изображения
@router.message(UserStates.generate_img)
async def generate_img(message: types.Message, state: FSMContext) -> None:
    user_data = await state.get_data()
    await message.answer(
        text="Запущена генерация изображения.\n\n"
        "Пожалуйста, подожите несколько минут..."
    )

    # зададим путь по которому будет сохраняться результат преобразования
    result_image_path = os.path.join(
        temp_dir, f"result_{message.message_id}.jpg"
    )

    content_image_path = user_data["content_img_file_name"]

    if user_data["model"] == "CGAN":
        model = Model_style_transfer_CGAN(user_data['cgan_style_name'])
        model.get_style_transfer_image(content_image_path, result_image_path)
    else:
        style_image_path = user_data["style_img_file_name"]
        model = Model_style_transfer_gram()
        model.get_style_transfer_image(
            content_image_path, style_image_path, result_image_path
        )

    # Отправка файла из файловой системы
    result_image = types.FSInputFile(result_image_path)
    await message.answer_photo(result_image, caption="Результат")

    # Устанавливаем и возвращаемся в начальное состояние
    await state.clear()
    await state.set_state(UserStates.start)
    await start_state(message, state)
