# Use the Python 3.11 base image with Alpine Linux
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel


# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install system dependencies

# Upgrade pip
RUN pip install --upgrade pip
# Install system dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install PyTorch and torchvision
CMD python app.py