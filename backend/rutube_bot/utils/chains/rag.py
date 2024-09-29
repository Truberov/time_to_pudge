from operator import itemgetter
from typing import List, Dict, Any, Optional

from langchain.retrievers import MultiVectorRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.storage import RedisStore
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import AsyncOpenAI

from rutube_bot.config.utils import get_settings

COLLECTION_NAME = "rutube_base"
PERSIST_DIR = "./chroma_data"
MODEL_NAME = "rscr/vikhr_nemo_12b:latest"
ID_KEY = "doc_id"
TOP_K = 30

ANSWER_SYSTEM_PROMPT = """\
Your task is to provide a clear, concise, and professional answer to \
the user's question only based on the provided information for the RUTUBE video hosting platform. \
Give two answers to each question: one with a list of relevant document identifiers and the second \
with the answer to the question itself, using documents with these identifiers."""

ANSWER_HUMAN_PROMPT = """\
<instuction>
Here is the instruction:

(1) Analyze all the provided Answers from the knowledge base and break them down into a set of facts. 
(2) Take each fact from the set and evaluate whether the taken fact answers the user's question. 
(3) If there are no such facts, answer only "Я не знаю". 
(4) Select facts that answer the user's question and are logically related to avoid loss of meaning in the answer to the user's question. 
(5) Compile the selected facts into a logical and structured text.
</instuction>
<attantion>
IMPORTANT: Your goal is to provide answers only within the scope of the RUTUBE video \
hosting platform and to provide accurate information to solve the user's problem. \
Use all relevant facts from the knowledge base. \
If there are no answers to the user's question in the found facts, say only "Я не знаю".
<attantion>
<input>
    {question}
</input>"""

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
    import json

    docs = answer_data.get("docs", [])
    docs_relevant_indexes = answer_data.get('relevant_indexes')

    try:
        docs_relevant_indexes = json.loads(docs_relevant_indexes)
        doc_metadata = docs[docs_relevant_indexes["relevant_doc_ids"][0]].metadata
        docs = [docs[index] for index in docs_relevant_indexes["relevant_doc_ids"]]
    except BaseException:
        doc_metadata = docs[0].metadata if docs else {}

    return {
        "class_1": doc_metadata.get('class_1'),
        "class_2": doc_metadata.get('class_2'),
        "answer": answer_data["answer"],
        "total_docs": len(docs),
        "docs": [doc.metadata.get('answer') for doc in docs],
    }


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
        return documents[:TOP_K]


def get_retriever() -> BaseRetriever:
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
    slice_compressor = SliceDocumentCompressor()
    pipeline = DocumentCompressorPipeline(transformers=[slice_compressor])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=multi_retriever
    )
    return compression_retriever


def create_retriever_chain(_retriever: BaseRetriever) -> Runnable:
    return itemgetter('question') | _retriever


def format_docs_for_vikhr(docs: List[Document]) -> str:
    import json

    docs = [{
        "doc_id": index,
        "title": doc.metadata.get('question'),
        "content": doc.metadata.get('answer')
    } for index, doc in enumerate(docs)]

    return json.dumps(docs, ensure_ascii=False)


async def create_chat_completions(_input: Dict[str, Any]) -> str:
    client = AsyncOpenAI(base_url=settings.LLM_URL, api_key='rutube')
    question = _input['question']
    docs = _input['context']
    relevant_indexes = _input.get('relevant_indexes')
    prompt = [
        {'role': 'system', 'content': ANSWER_SYSTEM_PROMPT},
        {'role': 'documents', 'content': docs},
        {'role': 'user', 'content': ANSWER_HUMAN_PROMPT.format(question=question)}
    ]

    if relevant_indexes:
        prompt.append({'role': 'assistant', 'content': relevant_indexes})

    llm_model = 'Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24'
    res = (await client.chat.completions.create(
        model=llm_model,
        messages=prompt,
        temperature=0.25,
        max_tokens=2048
    )).choices[0].message.content
    return res


def create_base_chain(
        _retriever: BaseRetriever
) -> Runnable:
    retriever_chain = create_retriever_chain(_retriever)

    context = (
        RunnablePassthrough.assign(
            docs=retriever_chain
        ).assign(context=lambda x: format_docs_for_vikhr(x["docs"]))
    )

    return context.assign(
        relevant_indexes=RunnableLambda(create_chat_completions)
    ).assign(answer=RunnableLambda(create_chat_completions))


def create_chain_with_formated_output(_base_chain: Runnable) -> Runnable:
    return _base_chain.pick(["answer", "docs", "relevant_indexes"]) | format_answer


retriever = get_retriever()
base_chain = create_base_chain(retriever)
chain_with_formated_output = create_chain_with_formated_output(base_chain)
