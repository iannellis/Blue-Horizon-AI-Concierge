FROM python:3.13-slim

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of
# application code changes.
COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY blue_horizon/ ./blue_horizon/
COPY ui/ ./ui/

# Only the two pkl files needed at runtime (rooms SQL reload feature).
# The rest of data/pandas/ is excluded via .dockerignore.
RUN mkdir -p data/pandas
COPY data/pandas/rooms.pkl data/pandas/room_availability.pkl ./data/pandas/

# Make blue_horizon importable without installing it as a package.
ENV PYTHONPATH=/app

# ── Deployment configuration ───────────────────────────────────────────────────
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/supervisord.conf /etc/supervisor/supervisord.conf
COPY deploy/.streamlit/ ./.streamlit/

# HuggingFace Spaces exposes exactly one port.
EXPOSE 7860

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
