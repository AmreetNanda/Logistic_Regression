FROM python:3.10-slim

# Set working directory
WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Streamlit needs this to allow container execution
ENV STREAMLIT_HOME=/app

# Run streamlit when container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]