"""Local NVIDIA Docker adapter for Harbor 0.23's Docker environment.

Harbor's stock Docker provider reports no GPU allocation capability, even
when the host Docker daemon supports NVIDIA devices. Use this provider with
gpu-compose.yaml on a host verified with ``docker run --gpus all``.
"""

from harbor.environments.docker.docker import DockerEnvironment


class DockerGPUEnvironment(DockerEnvironment):
    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"gpus": True})
