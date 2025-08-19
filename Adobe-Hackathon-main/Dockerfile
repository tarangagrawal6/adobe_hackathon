# Stage 1: Python Dependencies Builder
FROM python:3.11 AS python-builder
WORKDIR /app
# Copy Python dependency files
COPY requirements.txt ./
# Install Python dependencies with CPU-only PyTorch index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# Install Playwright browsers
RUN playwright install chromium
# Copy the download script
COPY download_models.py ./
# Download models during build
RUN python download_models.py

# Stage 2: Node.js Builder
FROM node:18-slim AS node-builder
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm install --no-cache
COPY client .
RUN npm run build

# Stage 3: Final Runtime Image
FROM python:3.11-slim
WORKDIR /app
# Install runtime dependencies including Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    ca-certificates \
    # Playwright system dependencies
    libnspr4 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libdrm2 \
    libgbm1 \
    libasound2 \
    libgtk-3-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=python-builder /usr/local/lib/python3.11/ /usr/local/lib/python3.11/
COPY --from=python-builder /usr/local/bin/ /usr/local/bin/
# Copy downloaded models from builder
COPY --from=python-builder /app/models/ /app/models/
# Copy Playwright browsers
COPY --from=python-builder /root/.cache/ms-playwright /root/.cache/ms-playwright
# Copy frontend build
COPY --from=node-builder /app/client/dist /app/client/dist
# Copy application code
COPY . .
# Create uploads directory
RUN mkdir -p uploads
# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
# Expose port
EXPOSE 8080
# Run the application
CMD ["python", "main.py"]
