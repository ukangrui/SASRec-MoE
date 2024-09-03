cd into main dir
podman build -t moelorapod:latest .
podman run -it \
  -v /home/ky2390/MoELoRA_Exp/main/:/moelora/ \
  --device nvidia.com/gpu=1 \
  --security-opt=label=disable \
  localhost/moelorapod:latest /bin/sh



podman run -it \
-v /home/ky2390/MoELoRA_Exp/main/:/moelora/ \
--security-opt=label=disable \
localhost/moelorapod:latest /bin/sh