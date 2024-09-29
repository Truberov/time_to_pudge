![kandinsky-download-1727581197737](https://github.com/user-attachments/assets/e67ce54a-5cb6-424f-a451-d4b1c9e36938)

 # :robot: AI-assistent for help desk 

### Команда AiRina предствляет решение на кейс Интеллектуальный помощник оператора службы поддержки 
[Ссылка на решение](https://t.me/airina_rutube_bot)


## :exploding_head: Проблематика

Видеоплатформа RUTUBE ежедневно получает тысячи запросов от пользователей по разным вопросам, связанным с использованием сервиса. 
В настоящее время обработка каждого такого запроса выполняется вручную операторами службы поддержки, что приводит к значительным затратам времени и ресурсов.

## :hugs: Решение

Решением этой задачи станет AI-ассистент, который поможет операторам службы поддержки быстро находить ответы, основываясь на базе знаний RUTUBE. Интеллектуальный 
помощник значительно ускорит работу операторов, что позволит сэкономить значительное количество человеко-часов и обеспечит пользователям оперативные ответы на их 
обращения в техподдержку, делая сервис еще более привлекательным.

## :building_construction: Архитектра решения

Интеллектуальный чат-бот построен на RAG-pipeline, который включает в себя:
- Bi-encoder
- Cross-encoder
- LLM

### :pencil2: Ввод пользователя и oбработка запроса

Происходит нормализация и удаление стоп слов для уменьшение шума на эмбеддингах

> [!Note] 
> Мы посчитали наиболее часто встречающиеся слова на реальных данных (например здравствуйте, спасибо и т.д.)
> и включили их в стоп слова, что положительное повлияло на метрику нахождения релевантых ответов из БЗ

### :mag_right: Поиск запроса по ответам БЗ

Векторным поиском находим топ 30 релеваных документов 

> [!Note]
> Для наибольшей релевантности выполняем поиск не только по собственным векторам вопросам из БЗ, а так же по перефразированным вопросам
>

### :bookmark_tabs: Фильтруем документы

LLM выполняет фильтрацию документов и выбирает k документов для генерации ответа

>[!Note]
>Для наибольшей релевантности выполняем поиск не только по собственным веторам ответов из БЗ, а так же по перефразированным ответа
>

### :bulb: Генерация ответа

На основании отфильтрованных документов, генерируем ответ
> [!Note]
> Но если с фильтрации пришло 0 документов, модель сразу ответит "Я не знаю". Поэтому у модели нет возможности
> придумывать свои ответ. Ответы всегда будут основываться на БЗ
>

 # :rocket: Запуск
Решение упаковано и будет готов к работе через **2 строки**

 **Для запуска нужны**
 - docker
 - docker-compose
 - make

**Запуск инференса LLM**
```
vllm serve --dtype half --max-model-len 16000 -tp 1 Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24 --api-key token-abc123
```
   
**Развертывание**
```
make up
```

 # :computer: Стек технологий
**БД**
- redis (хранилище связей аугментированных вопросов и вопрос из бз)
- chroma (векторное хранилище)

**Код**
- python
- langchain

# :checkered_flag: Итог
**В конечном итоге мы предлагалем ready-to-start решение у которого**

:heavy_check_mark: Быстрый инференс (до 20 секнуд) 

:heavy_check_mark: Точнось и полнота ответа 0.85

:heavy_check_mark: Точность подбора документов 0.94 в топ k отобранных фильтром

:heavy_check_mark: Высокая степень защиты от галлюцинаций

:heavy_check_mark: Легкая интеграция


## made with ♥️ by AiRina for 
![header-logo c7e8f395](https://github.com/user-attachments/assets/8a56ca15-e17a-4ab6-b864-017fce804610)



