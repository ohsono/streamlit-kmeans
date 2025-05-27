FROM python:3.12

WORKDIR /app

COPY . /app
 
RUN pip install --upgrade -r /app/requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]




