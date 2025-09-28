#use official Python Image
FROM python:3.10-slim

#set working directory in the container

WORKDIR /app

#copy the requirements first (for caching)
COPY requirements.txt .

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the files

COPY . .

#expose the port FastAPI runs on
EXPOSE 8000

#Run the API with uvicorn
CMD ["uvicorn","api:app", "--host","0.0.0.0","--port","8000"]