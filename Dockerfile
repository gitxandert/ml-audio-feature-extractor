ARG PY_TAG=3.13-slim-bullseye         # or 3.13-slim once Docker is updated
FROM python:${PY_TAG} as base

ENV DEBIAN_FRONTEND=noninteractive

# ---- system deps (needed at runtime too) ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---- build wheels --------------------------------------------
FROM base AS build
WORKDIR /tmp
COPY requirements.txt requirements-dev.txt /tmp/
ENV PIP_CACHE_DIR=/tmp/pip-cache
RUN pip install --upgrade pip && \
    pip wheel --cache-dir=$PIP_CACHE_DIR --wheel-dir /wheels \
        -r requirements.txt \
        -r requirements-dev.txt && \
    rm requirements*.txt


# ---- final image ---------------------------------------------
FROM base
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --cache-dir=$PIP_CACHE_DIR /wheels/*.whl
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