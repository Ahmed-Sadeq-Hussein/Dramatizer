# docker/app.Dockerfile | dramatizer: the project's code on top of dramatizer-base. A few MB; built on every
# push to main by CI (job "image"), or by hand:
#
#   docker build -f docker/app.Dockerfile --build-arg BASE_IMAGE=<base@sha256:...> --build-arg GIT_COMMIT=<sha> -t dramatizer:dev .
#
# What may be copied in is decided by .dockerignore (an allow list). Run it with deploy/container.env and the
# mounts in Milestone 4.md, step 2:
#
#   docker run --gpus all --env-file deploy/container.env -p 127.0.0.1:5005:5005 \
#     -v dramatizer-state:/app/history -v "D:/Dramatizer 2 renders:/output" -v "D:/Dramatizer models:/models:ro" ...

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG GIT_COMMIT=unknown

# Not root: a hole in the server is then a hole as an ordinary user, inside the container
RUN useradd --create-home --uid 10001 dramatizer
WORKDIR /app
# The code belongs to root: the server's user can read it but not change it
COPY . /app
# history/ holds the database, accounts, API keys, the TLS certificate and logs. It is a Docker named volume
# (dramatizer-state) at run time; creating it here, owned by the user, makes a new volume start out writable.
RUN mkdir -p /app/history/api && chown -R dramatizer:dramatizer /app/history
USER dramatizer

ENV GIT_COMMIT=${GIT_COMMIT}
LABEL org.opencontainers.image.source="https://github.com/Ahmed-Sadeq-Hussein/Dramatizer" \
      org.opencontainers.image.revision="${GIT_COMMIT}"
EXPOSE 5005
VOLUME ["/app/history"]
CMD ["python", "main_backend.py", "api"]
