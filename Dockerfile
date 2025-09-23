# --- Stage 1: The Builder ---
# Use an official Python image as a parent image
FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /app

# Prevent Poetry from creating a virtual environment inside the container
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install Poetry
RUN pip install poetry

# Copy the dependency files to the container
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-root --only main

# --- Stage 2: The Runner ---
# Use a fresh, lightweight Python image for the final container
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app

# Copy the installed dependencies from the builder stage
COPY --from=builder /app /app

# Copy the application source code
COPY src/llm_job_assistant ./src/llm_job_assistant

# Switch to the non-root user
USER app

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run when the container starts
CMD ["poetry", "run", "start"]
