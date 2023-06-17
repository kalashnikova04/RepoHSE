Инструкция по установке


Ссылка на образ на докерхабе:
https://hub.docker.com/layers/ily0ff/hse_project/latest/images/sha256:26a2d5d0a36aed43d39f0221b95817942a602d9cdfb3b1c21e740d93bd43aba4


1. Базовый образ: python:3.9
2. Рабочая директория: /app
3. Копирование файла requirements.txt в контейнер
4. Установка зависимостей командой "pip install --no-cache-dir -r requirements.txt"
5. Копирование файлов main.py и model.pickle в контейнер
6. Открытие порта 8000 для приложения через инструкцию EXPOSE
7. Запуск приложения с помощью команды "uvicorn --host 0.0.0.0 --port 8898 main:app"
8. Этот образ запускает приложение на базе фреймворка FastAPI, которое выполняет прогноз по предоставленному тексту, используя модель машинного обучения, сохраненную в файле model.pickle.


Чтобы установить необходимо выполнить следующие команды:


```docker pull ily0ff/hse_project:latest```


```docker run -p 8000:8898 ily0ff/hse_project:latest```


Сервис будет запущен на 0.0.0.0:8000
