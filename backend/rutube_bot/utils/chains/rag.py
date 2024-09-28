from operator import itemgetter
from typing import List, Dict, Any, Optional

from langchain.output_parsers import BooleanOutputParser
from langchain.retrievers import MultiVectorRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, DocumentCompressorPipeline
from langchain_community.storage import RedisStore
from langchain_core.callbacks import Callbacks
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from rutube_bot.config.utils import get_settings

COLLECTION_NAME = "rutube_base"
PERSIST_DIR = "./chroma_data"
MODEL_NAME = "rscr/vikhr_nemo_12b:latest"
ID_KEY = "doc_id"

FILTER_DOCS_PROMPT = """\
Ваша задача - определить, соответствует ли предоставленная информация из базы знаний вопросу пользователя.
Входные данные:

Вопрос пользователя
Вопрос из базы знаний
Ответ из базы знаний

Инструкции:

Внимательно прочитайте вопрос пользователя.
Проанализируйте вопрос и ответ из базы знаний.
Оцените, насколько полно информация из базы знаний отвечает на вопрос пользователя.
Вынесите решение:

Если информация из базы знаний полностью (на 100%) отвечает на вопрос пользователя, ответьте "YES".
Если информация из базы знаний не полностью отвечает на вопрос пользователя или не относится к нему, ответьте "NO".

Формат ответа:
Ответьте только "YES" или "NO" без дополнительных пояснений.
Примечания:

Обратите особое внимание на ключевые слова и основную суть вопроса пользователя.
Учитывайте, что формулировки вопросов могут отличаться, но смысл может быть одинаковым.
Если есть хотя бы малейшее сомнение в полноте ответа, выберите "NO".

<context>
    {context} 
<context/>

<question>
    {question} 
<question/>"""

ANSWER_SYSTEM_PROMPT = """\
###PREAMBLE###
ВЫ - ВЫСОКОКВАЛИФИЦИРОВАННЫЙ АССИСТЕНТ СЛУЖБЫ ПОДДЕРЖКИ ВИДЕОХОСТИНГА RUTUBE. ВАША ЗАДАЧА - ПРЕДОСТАВЛЯТЬ ТОЧНЫЕ, КРАТКИЕ И ИНФОРМАТИВНЫЕ ОТВЕТЫ НА ВОПРОСЫ ПОЛЬЗОВАТЕЛЕЙ, ОСНОВЫВАЯСЬ ИСКЛЮЧИТЕЛЬНО НА ПРЕДОСТАВЛЕННОМ КОНТЕКСТЕ.

###ИНСТРУКЦИИ###
- ВНИМАТЕЛЬНО АНАЛИЗИРУЙТЕ ВОПРОС ПОЛЬЗОВАТЕЛЯ И ПРЕДОСТАВЛЕННЫЙ КОНТЕКСТ.
- ОТВЕЧАЙТЕ ТОЛЬКО НА ОСНОВЕ ИНФОРМАЦИИ ИЗ КОНТЕКСТА, НЕ ДОБАВЛЯЯ НИЧЕГО ОТ СЕБЯ.
- ЕСЛИ В КОНТЕКСТЕ НЕТ ИНФОРМАЦИИ ДЛЯ ОТВЕТА, ЧЕТКО СКАЖИТЕ "Я не знаю".
- ГДЕ ВОЗМОЖНО, ДАВАЙТЕ КРАТКИЙ И ЧЕТКИЙ ОТВЕТ.
- ЕСЛИ ТРЕБУЕТСЯ ПОЛНЫЙ ОТВЕТ, ПРЕДОСТАВЬТЕ ЕГО, НО СТРОГО В РАМКАХ КОНТЕКСТА.
- НЕ ИСПОЛЬЗУЙТЕ ФРАЗЫ ВРОДЕ "На основе предоставленной информации" ИЛИ "Согласно контексту".
- НЕ ИЗВИНЯЙТЕСЬ И НЕ ИСПОЛЬЗУЙТЕ ИЗЛИШНЕ ВЕЖЛИВЫЕ ФОРМУЛИРОВКИ.

СЛЕДУЙТЕ ЭТОЙ ЦЕПОЧКЕ МЫСЛЕЙ:

1. **АНАЛИЗ ЗАПРОСА:**
   1.1 ВНИМАТЕЛЬНО ПРОЧИТАЙТЕ ВОПРОС ПОЛЬЗОВАТЕЛЯ.
   1.2 ОПРЕДЕЛИТЕ КЛЮЧЕВЫЕ АСПЕКТЫ ЗАПРОСА.

2. **АНАЛИЗ КОНТЕКСТА:**
   2.1 ИЗУЧИТЕ ПРЕДОСТАВЛЕННЫЙ КОНТЕКСТ.
   2.2 НАЙДИТЕ РЕЛЕВАНТНУЮ ИНФОРМАЦИЮ ДЛЯ ОТВЕТА.

3. **ФОРМУЛИРОВКА ОТВЕТА:**
   3.1 ЕСЛИ ИНФОРМАЦИИ НЕДОСТАТОЧНО, ОТВЕТЬТЕ "Я не знаю".
   3.2 ЕСЛИ ВОЗМОЖЕН КРАТКИЙ ОТВЕТ, СФОРМУЛИРУЙТЕ ЕГО ЧЕТКО И ЛАКОНИЧНО.
   3.3 ЕСЛИ НУЖЕН ПОЛНЫЙ ОТВЕТ, ПРЕДОСТАВЬТЕ ЕГО, СТРОГО ПРИДЕРЖИВАЯСЬ КОНТЕКСТА.

4. **ПРОВЕРКА:**
   4.1 УБЕДИТЕСЬ, ЧТО ОТВЕТ ТОЧНО СООТВЕТСТВУЕТ ВОПРОСУ И КОНТЕКСТУ.
   4.2 ПРОВЕРЬТЕ, НЕТ ЛИ В ОТВЕТЕ ИНФОРМАЦИИ, ОТСУТСТВУЮЩЕЙ В КОНТЕКСТЕ.

### ЧЕГО НЕ ДЕЛАТЬ ###
1. **НЕ ДОБАВЛЯЙТЕ ИНФОРМАЦИЮ, КОТОРОЙ НЕТ В КОНТЕКСТЕ.**
2. **НЕ ИСПОЛЬЗУЙТЕ ОБЩИЕ ФРАЗЫ ИЛИ ПРЕДПОЛОЖЕНИЯ.**
3. **НЕ ДАВАЙТЕ НЕПОЛНЫЕ ИЛИ НЕТОЧНЫЕ ОТВЕТЫ.**
4. **НЕ ИГНОРИРУЙТЕ ЧАСТИ ВОПРОСА ПОЛЬЗОВАТЕЛЯ.**
5. **НЕ ПРЕДЛАГАЙТЕ ДОПОЛНИТЕЛЬНУЮ ПОМОЩЬ ИЛИ КОНТАКТЫ ПОДДЕРЖКИ, ЕСЛИ ОБ ЭТОМ НЕ СПРАШИВАЮТ.**

ПОМНИТЕ: ВАША ГЛАВНАЯ ЦЕЛЬ - ПРЕДОСТАВИТЬ ТОЧНЫЙ, КРАТКИЙ И ИНФОРМАТИВНЫЙ ОТВЕТ, СТРОГО СООТВЕТСТВУЮЩИЙ ВОПРОСУ И КОНТЕКСТУ.

<context>
    {context}
<context/>
"""
FILTER_MERGED_DOCUMENTS_PROMPT = """Ваша роль - анализировать вопрос пользователя и сравнивать \
его с предоставленными вопросами и ответами из базы знаний. \
Ваша цель - определить, какой из представленных ответов наиболее подходит для вопроса пользователя."""

HUMAN_FILTER_MERGED_DOCUMENTS_PROMPT = """\
Инструкции:
1. Внимательно прочитайте вопрос пользователя.
2. Проанализируйте каждый вопрос и ответ из предоставленной базы знаний.
3. Определите, какой ответ наиболее точно и полно соответствует вопросу пользователя.
4. Если найден подходящий ответ, верните только номер соответствующего документа.
5. Если ни один из представленных ответов не подходит, верните -1.
6. Ответ должен быть строго в формате JSON, без дополнительных пояснений или текста.

База знаний:
<context>
{context}
</context>

Вопрос пользователя:
<question>
{question}
</question>

Формат ответа (JSON):
{{
    "doc_index": целое_число
}}

Примеры корректных ответов:
{{
    "doc_index": 2
}}
или
{{
    "doc_index": -1
}}

Важно: 
- Ответ должен содержать только JSON с номером документа или -1, если подходящего ответа нет.
- Не добавляйте никаких дополнительных слов, знаков препинания или пояснений.
- Убедитесь, что ваш ответ является валидным JSON."""

DEFAULT_ANSWER_FOR_UNSUITABLE_QUESTIONS = """\
Извините, не могу вам помочь, я не знаю ответ на этот вопрос."""

settings = get_settings()


def get_embeddings_model() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")


def format_docs(docs: List[Document]) -> str:
    """Return the formatted doc for chain input."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        question = doc.metadata.get('question')
        answer = doc.metadata.get('answer')
        doc_string = f"№{i} Вопрос из базы знаний: {question}\nОтвет из базы знаний: {answer}"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def format_answer(answer_data: Dict[str, Any]) -> Dict[str, Any]:
    docs = answer_data.get("docs", [])
    doc_metadata = docs[0].metadata if docs else {}

    return {
        "class_1": doc_metadata.get('class_1'),
        "class_2": doc_metadata.get('class_2'),
        "answer": answer_data["answer"],
        "total_docs": len(answer_data["docs"]),
        "docs": [doc.metadata.get('answer') for doc in docs],
    }


def get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": format_docs([doc])}


class SliceDocumentCompressor(BaseDocumentCompressor):
    def compress_documents(
            self,
            documents: List[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.
        """
        return documents[:20]


def get_retriever(_model) -> BaseRetriever:
    embeddings_model = get_embeddings_model()
    vectorstore = Chroma(
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )
    byte_store = RedisStore(redis_url=settings.REDIS_URL)
    multi_retriever = MultiVectorRetriever(
        id_key=ID_KEY,
        byte_store=byte_store,
        vectorstore=vectorstore
    )
    multi_retriever.search_kwargs = dict(k=1000)
    prompt = PromptTemplate(
        template=FILTER_DOCS_PROMPT,
        input_variables=["question", "context"],
        output_parser=BooleanOutputParser(),
    )

    llm_filter = LLMChainFilter.from_llm(_model, prompt=prompt, get_input=get_input)
    slice_compressor = SliceDocumentCompressor()
    pipeline = DocumentCompressorPipeline(transformers=[slice_compressor, llm_filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=multi_retriever
    )
    return compression_retriever


def create_retriever_chain(_retriever: BaseRetriever, _model: LanguageModelLike) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ('system', FILTER_MERGED_DOCUMENTS_PROMPT),
        ('human', HUMAN_FILTER_MERGED_DOCUMENTS_PROMPT)
    ])

    def get_doc_by_index(_input: Dict[str, Any]) -> List[Document]:
        doc_index = _input["filtered_doc"].get("doc_index", -1)
        docs = _input["docs"]
        if doc_index != -1:
            return [docs[doc_index]]
        return []

    pick_document_chain = {
                              "context": RunnableLambda(lambda x: x["docs"]) | format_docs,
                              "question": itemgetter('question')
                          } | prompt | _model | JsonOutputParser()

    filter_documents_chain = {
                                 "docs": itemgetter('docs'),
                                 "filtered_doc": pick_document_chain
                             } | RunnableLambda(get_doc_by_index)

    def check_num_documents(_input: Dict[str, Any]) -> Runnable:
        docs = _input.get('docs')
        if len(docs) > 1:
            return filter_documents_chain

        return docs

    # return {
    #     "question": itemgetter('question'),
    #     "docs": itemgetter('question') | _retriever
    # } | RunnableLambda(check_num_documents)

    return itemgetter('question') | _retriever


def create_base_chain(
        _model: LanguageModelLike,
        _retriever: BaseRetriever
) -> Runnable:  # TODO: Добавить случай определения классов, если документы не доехали
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    retriever_chain = create_retriever_chain(_retriever, _model)

    context = (
        RunnablePassthrough.assign(
            docs=retriever_chain
        ).assign(context=lambda x: format_docs(x["docs"]))
    )

    response_synthesizer = prompt | _model | StrOutputParser()

    def check_num_documents(_input: Dict[str, Any]) -> Runnable | str:
        docs = _input["docs"]
        if len(docs) != 0:
            return response_synthesizer

        return DEFAULT_ANSWER_FOR_UNSUITABLE_QUESTIONS

    answer_chain = RunnableLambda(check_num_documents)

    return context.assign(answer=answer_chain)


def create_chain_with_formated_output(_base_chain: Runnable) -> Runnable:
    return _base_chain.pick(["answer", "docs"]) | format_answer


model = ChatOllama(model=MODEL_NAME, base_url=settings.OLLAMA_URL)
retriever = get_retriever(model)
base_chain = create_base_chain(model, retriever)
chain_with_formated_output = create_chain_with_formated_output(base_chain)
