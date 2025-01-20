
# Better SASRec  

**Better SASRec** is a cleaned-up and optimized version of utilities for **SASRec (Self-Attentive Sequential Recommendation)**, based on the original implementation. It improves code readability and modularity for easier use in recommendation tasks.

## Features  
- Refactored utilities for SASRec  
- Improved evaluation efficiency by introducing batch processing
- Compatible with the original implementation  


## Acknowledgment  
This project is based on the original implementation:  
**[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)**  


cd into main dir
podman build -t moelorapod:latest .
podman run -it \
  -v /home/ky2390/MoELoRA_Exp/main/:/moelora/ \
  --device nvidia.com/gpu=0 \
  --security-opt=label=disable \
  localhost/moelorapod:latest /bin/sh



podman run -it \
-v /home/ky2390/MoELoRA_Exp/main/:/moelora/ \
--security-opt=label=disable \
localhost/moelorapod:latest /bin/sh













base model      ndcg: 0.10812802472545699, ht: 0.23427152317880795
lora hard model ndcg: 0.11075738290534327, ht: 0.23956953642384105



1. predefined groups, predefined tasks, frozen    lora -> ndcg: 0.1085467725641643, ht: 0.23427152317880795
2. learnt     groups, predefined tasks, trainable lora -> ndcg: 0.10820832858447192
3. learnt     groups, learnt     tasks, trainable lora -> ndcg: 0.1082337573154062





1/7

MoE Hard ndcg: 0.110757382905343, ht: 0.23956953642384107