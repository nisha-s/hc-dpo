# CPU image for running the tests (and the debug config) locally. Real runs use Colab.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
