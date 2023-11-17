ARG REGISTRY="nvcr.io"
ARG CUDA_VERSION="11.8.0"
FROM mambaorg/micromamba:1.5.1 as micromamba
FROM ${REGISTRY}/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

#####
# Setup micromamba
#####

USER root

ARG MAMBA_USER=yi
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=100
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV ENV_NAME=yi

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="Asia/Shanghai"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y \
  && apt-get install -y sudo tzdata git ninja-build \
  && useradd -ms /bin/bash -d /home/$MAMBA_USER $MAMBA_USER --uid $MAMBA_USER_ID --gid $MAMBA_USER_GID \
  && usermod -aG sudo $MAMBA_USER \
  && echo "$MAMBA_USER ALL=NOPASSWD: ALL" >> /etc/sudoers \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

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

WORKDIR /home/${MAMBA_USER}/workspace/Yi
COPY --chown=${MAMBA_USER_ID}:${MAMBA_USER_GID} ./conda-lock.yml .
RUN micromamba create -y -n ${ENV_NAME} -f conda-lock.yml && \
  micromamba clean --all --yes

COPY --chown=${MAMBA_USER_ID}:${MAMBA_USER_GID} . .