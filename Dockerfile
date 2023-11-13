ARG REGISTRY="nvcr.io"
ARG CUDA_VERSION="11.8.0"
FROM mambaorg/micromamba:1.5.1 as micromamba
FROM ${REGISTRY}/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

#####
# Setup user & common tools
#####
RUN apt update \
  && apt install -y git ninja-build \
  && rm -rf /var/lib/apt/lists/*

#####
# Setup micromamba
#####

USER root

ARG MAMBA_USER=yi
ARG MAMBA_USER_ID=56789
ARG MAMBA_USER_GID=56789
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV ENV_NAME=yi

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
  /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]

# Install dependencies

WORKDIR /yi
COPY ./pyproject.toml .
RUN micromamba create -y -n ${ENV_NAME} -f pyproject.yml && \
  micromamba clean --all --yes

COPY . .