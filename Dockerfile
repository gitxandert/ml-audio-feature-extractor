ARG PY_TAG=3.10-slim
FROM python:${PY_TAG} as base

ENV DEBIAN_FRONTEND=noninteractive

# ---- system deps (needed at runtime too) ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

FROM base
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install "numpy<2.0" && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
COPY . .
CMD ["python", "main.py"]

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y --no-install-recommends sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME/.cache \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

USER $USERNAME
ENV HOME=/home/$USERNAME
# WORKDIR /app
