FROM python:3.10.12

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /agent_bb7

COPY . .

RUN apt-get update && apt-get install -y postgresql-client && apt-get install -y dnsutils iputils-ping vim

CMD ["python", "agent_bb7/main.py"]