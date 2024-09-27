import asyncio
import logging
import sys
import os
import aiohttp

from aiogram import Bot, Dispatcher, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

BOT_TOKEN = os.environ.get('TG_TOKEN')
MICROSERVICE_URL = os.environ.get('RAG_URL')

dp = Dispatcher()


async def get_answer_from_microservice(question):
    async with aiohttp.ClientSession() as session:
        async with session.post(MICROSERVICE_URL, json={'question': question}) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('answer', 'Извините, не удалось получить ответ.')
            else:
                return 'Извините, произошла ошибка при обращении к сервису.'


@dp.message(CommandStart())
async def send_welcome(message: types.Message) -> None:
    await message.reply(
        "Привет! Я бот, который может ответить на ваши вопросы. \
        Просто напишите свой вопрос, и я постараюсь на него ответить.")


@dp.message()
async def answer_question(message: types.Message) -> None:
    question = message.text
    answer = await get_answer_from_microservice(question)
    await message.answer(answer)


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
