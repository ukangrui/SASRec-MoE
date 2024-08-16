cd into main dir
podman build -t moelorapod:latest .
podman run -it \
  -v /Users/kangrui/Desktop/MoELoRA/main/:/moelora/ \
  --device nvidia.com/gpu=00000000:37:00.0 \
  --security-opt=label=disable \
  localhost/moelorapod:latest /bin/sh