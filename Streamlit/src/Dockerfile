# Use an official Python runtime as a parent imageF
FROM python:3.9.5-slim-buster


# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the command to start the Streamlit app
CMD streamlit run main.py
