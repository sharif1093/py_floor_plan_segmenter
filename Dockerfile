FROM python:3.11.1-slim

ARG USERID=1000
ARG GROUPID=1000

ENV WORK_USER cloud
ENV USER_HOME /home/${WORK_USER}

# Set root password
# Install gosu
RUN echo "root:rootpass" | chpasswd \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get --no-install-recommends install -y \
  curl \
  git \
  gosu \
  jq \
  p7zip-full \
  graphviz \
  gcc \
  build-essential \
  ffmpeg \
  && apt-get clean all \
  && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -g ${GROUPID} -o ${WORK_USER} \
  && useradd -m -d ${USER_HOME} -u ${USERID} -g ${WORK_USER} ${WORK_USER} \
  && chown -R ${WORK_USER}:${WORK_USER} ${USER_HOME}

# Install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy files
COPY ./py_floor_plan_segmenter /app/py_floor_plan_segmenter
RUN chown -R ${WORK_USER}:${WORK_USER} /app

# Set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

WORKDIR ${USER_HOME}

EXPOSE 8008

CMD ["uvicorn", "py_floor_plan_segmenter.server:app", "--host", "0.0.0.0", "--port", "8008"]
