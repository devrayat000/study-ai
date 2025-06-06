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
RUN uv sync --frozen --no-cache --no-dev

# Set the environment variable for the app.
ENV DEBUG="false"
EXPOSE 7860

# Run the application.
CMD ["uv", "run", "main.py"]