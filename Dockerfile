FROM python:3.8
WORKDIR /root/wfhack2022
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pwd
COPY . .
RUN ls -ltr
ENTRYPOINT ["python","data_processor.py"]