FROM python:3.13-slim

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends git supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements and local wheels first so Docker can cache this layer
# independently of application code changes.
COPY deploy/requirements.txt ./
COPY assets/ ./assets/
RUN pip install --no-cache-dir assets/ml_dtypes-0.4.1-cp313-cp313-linux_x86_64.whl \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY blue_horizon/ ./blue_horizon/
COPY ui/ ./ui/

# Make blue_horizon importable without installing it as a package.
ENV PYTHONPATH=/app

# ── Deployment configuration ───────────────────────────────────────────────────
COPY deploy/supervisord.conf /etc/supervisor/supervisord.conf
COPY deploy/.streamlit/ ./.streamlit/

# HuggingFace Spaces exposes exactly one port.
EXPOSE 7860

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
