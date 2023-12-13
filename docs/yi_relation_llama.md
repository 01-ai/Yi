# Yi's relation with LLaMA

> 💡 TL;DR
> 
> The Yi series models adopt the same model architecture as LLaMA but are **NOT** derivatives of LLaMA.

- Both Yi and LLaMA are all based on the Transformer structure, which has been the standard architecture for large language models since 2018.

- Grounded in the Transformer architecture, LLaMA has become a new cornerstone for the majority of state-of-the-art open-source models due to its excellent stability, reliable convergence, and robust compatibility. This positions LLaMA as the recognized foundational framework for models including Yi.

- Thanks to the Transformer and LLaMA architectures, other models can leverage their power, reducing the effort required to build from scratch and enabling the utilization of the same tools within their ecosystems.

- However, the Yi series models are NOT derivatives of LLaMA, as they do not use LLaMA's weights.

  - As LLaMA's structure is employed by the majority of open-source models, the key factors of determining model performance are training datasets, training pipelines, and training infrastructure.

  - Developing in a unique and proprietary way, Yi has independently created its own high-quality training datasets, efficient training pipelines, and robust training infrastructure entirely from the ground up. This effort has led to excellent performance with Yi series models ranking just behind GPT4 and surpassing LLaMA on the [Alpaca Leaderboard in Dec 2023](https://tatsu-lab.github.io/alpaca_eval/). 
