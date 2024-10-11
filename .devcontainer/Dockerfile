# A unified parameterized dockerfile to be used for building devcontainers and lightweight
# python-based images to be used in production. The purpose is to replicate in production
# (as much as possible) the development environment setup.
ARG BASE_IMAGE
ARG TAG
ARG POETRY_VERSION
ARG POETRY_CORE_VERSION
ARG POETHEPOET_VERSION
ARG PRE_COMMIT_VERSION
ARG GIT_VERSION
ARG INSTALL_DEV_TOOLS
ARG INSTALL_DEV_TOOLS="no"
ARG INSTALL_GIT="no"


# ========================================================================================
# Production/development common initial stage
# ========================================================================================
FROM ${BASE_IMAGE}:${TAG} as git_setup_stage
ARG INSTALL_GIT
ARG GIT_VERSION
RUN if [ "${INSTALL_GIT}" != "no" ]; \
    then \
        apt-get update && \
        apt-get install --no-install-recommends -y git=${GIT_VERSION} && \
        git config --global --add safe.directory '*'; \
    fi

ARG POETRY_VERSION
ARG POETRY_CORE_VERSION
ARG POETHEPOET_VERSION
ARG PRE_COMMIT_VERSION
ARG INSTALL_DEV_TOOLS
RUN if [ "${INSTALL_DEV_TOOLS}" != "no" ]; \
    then \
        pip install poetry==${POETRY_VERSION} poetry-core==${POETRY_CORE_VERSION} poethepoet==${POETHEPOET_VERSION} pre-commit==${PRE_COMMIT_VERSION}; \
    fi
