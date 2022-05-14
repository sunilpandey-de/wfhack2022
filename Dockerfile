FROM python:3.7
WORKDIR /root/wfhack2022
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pwd
COPY . .
RUN ls -ltr
ENTRYPOINT ["python","data_processing/data_processor.py"]
