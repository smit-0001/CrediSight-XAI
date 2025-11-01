# Stage 1: Use an official Python slim image as the base
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy only the requirements file first
# This takes advantage of Docker's layer caching.
# If requirements.txt doesn't change, Docker won't reinstall packages.
COPY requirements.txt .

# Stage 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy all our application code and artifacts
# This includes the /api directory and the /artifacts directory
COPY ./api ./api
COPY ./artifacts ./artifacts

# Stage 6: Expose the port the app will run on
# This tells Docker the container listens on port 80
EXPOSE 80

# Stage 7: Define the command to run the application
# We use "0.0.0.0" to accept connections from outside the container
# We use port 80, the standard HTTP port
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]