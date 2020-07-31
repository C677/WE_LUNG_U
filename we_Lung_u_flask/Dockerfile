FROM python:3.7.8
LABEL maintainer "WLU"
RUN pip install --upgrade pip
COPY . /path
WORKDIR /path
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /path/
COPY templates/ /path/templates/
COPY static/ /path/static/
EXPOSE 80
CMD ["python", "./app.py"]