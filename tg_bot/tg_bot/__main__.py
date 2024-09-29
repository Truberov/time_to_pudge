import asyncio
import logging
import sys
import os
import aiohttp

from aiogram import Bot, Dispatcher, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.utils.markdown import bold, underline, code

BOT_TOKEN = os.environ.get('TG_TOKEN')
MICROSERVICE_URL = os.environ.get('RAG_URL')
DEFAULT_WAITING_MESSAGE = "Ваш запрос зарегестрирован. Скоро вернусь с ответом."

dp = Dispatcher()


async def get_answer_from_microservice(question: str) -> dict:
    """
    Makes request to microservice to get answer
    Args:
        question (str): user question
    Returns:
        response (dict): response containing answer and metadata
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(MICROSERVICE_URL, json={'question': question}) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                return {
                    'answer': 'Извините, произошла ошибка при обращении к сервису.',
                    'class_1': '',
                    'class_2': ''
                }


def format_answer(response: dict) -> str:
    """
    Formats the response from the microservice
    Args:
        response (dict): response from microservice
    Returns:
        formatted_answer (str): MarkdownV2 formatted answer
    """
    answer = response.get('answer', 'Извините, не удалось получить ответ.')
    class_1 = response.get('class_1', '')
    class_2 = response.get('class_2', '')

    # Экранируем специальные символы Markdown
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        answer = answer.replace(char, f'\\{char}')
        class_1 = class_1.replace(char, f'\\{char}')
        class_2 = class_2.replace(char, f'\\{char}')

    formatted_answer = (
        f"{answer}\n\n"
        f"{code(class_1.lower())} \\> {code(class_2.lower())}"
    )

    return formatted_answer


@dp.message(CommandStart())
async def send_welcome(message: types.Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.reply(
        "Привет! Я бот, который может ответить на ваши вопросы. \
        Просто напишите свой вопрос, и я постараюсь на него ответить.")


@dp.message()
async def answer_question(message: types.Message) -> None:
    """
    Handler will answer the user message with formatted answer from RAG service
    """
    question = message.text
    await message.answer(DEFAULT_WAITING_MESSAGE)
    response = await get_answer_from_microservice(question)
    formatted_answer = format_answer(response)
    await message.answer(formatted_answer, parse_mode="MarkdownV2", disable_web_page_preview=True)


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
