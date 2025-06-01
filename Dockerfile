# ============================
# 🏗️ Build Stage
# ============================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=2.1.3
ENV PATH="/root/.local/bin:$PATH"

# 必要なパッケージ（最小限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry インストール
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

WORKDIR /app

# Poetry 関連ファイルを先にコピーして依存解決
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi
# TODO 20250601: dev 依存は Cloud Run では不要なので除外したい
# RUN poetry install --without dev --no-root --no-interaction --no-ansi 

# freezeしてrequirements.txtを生成（runtime用）
RUN pip freeze > requirements.txt

# アプリケーションファイルをコピー（依存で不要なものが含まれないように分けている）
COPY src ./src
COPY data ./data

# ============================
# 🏃 Runtime Stage
# ============================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Python の実行に必要な最小限のパッケージのみ
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# builder から必要なファイルだけコピー
COPY --from=builder /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/src ./src
COPY --from=builder /app/data ./data

# 📦 コピーされたか確認
RUN ls -l /app/data

# 実行ポート（Cloud Run 用）
EXPOSE 8080

# アプリ起動
CMD ["gunicorn", "src.app:server", "--bind", "0.0.0.0:8080", "--workers", "1"]
