FROM python:3.11.9-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
# Install PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*
RUN uv lock
RUN uv sync --frozen --no-dev

# Set the environment variable for the app.
ENV DEBUG="false"
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

# Run the application.
CMD ["uv", "run", "main.py"]