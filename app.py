import os
from aiogram import Bot, Dispatcher
import asyncio
import logging

from handlers import default_commands, dialog

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=os.environ.get("TG_BOT_TOKEN"))

# Диспетчер
dp = Dispatcher()

# Подключаем обработчики событий
dp.include_router(default_commands.router)
dp.include_router(dialog.router)


async def main():
    # Удалим все накопленные события в очереди
    await bot.delete_webhook(drop_pending_updates=True)
    # Процесс постоянного опроса сервера на наличие новых событий от Telegram
    # При поступлении события, диспетчер вызовет первую(!) подходящую
    # по всем фильтрам ф-ю обработки
    print("Run bot")
    await dp.start_polling(bot)

# Запуск основного цикла программы
if __name__ == "__main__":
    asyncio.run(main())
