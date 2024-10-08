import random

from fastapi import APIRouter, status

from rutube_bot.schemas import QuestionRequest, RAGResponse
from rutube_bot.utils.chains import chain_with_formated_output

api_router = APIRouter(tags=["RAG"])


leha_answers = [
    "Очень интересный вопрос, боюсь ответ на него знает только Леха.",
    "Хм, отличный вопрос! Но, похоже, только Леха владеет этой тайной.",
    "Вот это да, вы задали загадку! Думаю, лишь Леха сможет ее разгадать.",
    "Занятно! Но, кажется, эту информацию хранит только Леха в своей голове.",
    "Ого, какой необычный вопрос! Боюсь, ответ на него - эксклюзив Лехи.",
    "Интригующе! Однако, похоже, что ключ к этой загадке есть только у Лехи.",
    "Вот это поворот! Но, увы, ответ на этот вопрос - монополия Лехи.",
    "Потрясающий вопрос! Жаль, что ответ на него - тайна, известная лишь Лехе.",
    "Вы затронули интересную тему! Но, кажется, только Леха может пролить свет на этот вопрос.",
    "Какой любопытный запрос! К сожалению, ответ на него, похоже, знает исключительно Леха.",
    "Вот это да, вы меня озадачили! Боюсь, придется обратиться к Лехе - похоже, только он владеет этой информацией."
]


@api_router.post(
    "/predict",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(question_data: QuestionRequest):
    answer = await chain_with_formated_output.ainvoke(question_data.dict())
    return RAGResponse(**answer)
