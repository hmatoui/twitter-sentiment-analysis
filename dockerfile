# Use the base image with Jupyter and TensorFlow for GPU support
# FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM tensorflow/tensorflow:latest-gpu

# Create and set the working directory
WORKDIR /app

# Copy the necessary files
COPY scripts /app/scripts
COPY data /app/data
COPY models /app/models
COPY requirements.txt /app

# Create a virtual environment
RUN python -m venv /venv

# Activate the virtual environment
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create the logs directory
RUN mkdir -p /app/logs

# Expose the Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
# Redirect logs to a file and stdout
CMD ["bash", "-c", "python scripts/train.py"]