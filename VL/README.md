## Quick Start 
python single_inference.py --model-path path-to-yi-vl-model --image-file path-to-image --question question-content
<img width="1043" alt="image" src="https://github.com/01-ai/Multimodal-LLaVA/assets/20470010/45077410-4a62-4384-96a9-325550ea44b5">


## Major difference with llava
1. We add laynorm in the two-layer mlp.
2. We train the parameters of ViT and scale up the input resolution.
3. We utilize laion400m data for pretraining.

