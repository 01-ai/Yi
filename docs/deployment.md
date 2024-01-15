# Hardware requirements

Before deploying Yi in your environment, make sure your hardware meets the following requirements.

## Chat models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|----------------------|--------------|:-------------------------------------:|
| Yi-6B-Chat           | 15 GB         | RTX 3090 <br> RTX 4090 <br>  A10 <br> A30             |
| Yi-6B-Chat-4bits     | 4 GB          | RTX 3060 <br>  RTX 4060                     |
| Yi-6B-Chat-8bits     | 8 GB          | RTX 3070 <br> RTX 4060                     |
| Yi-34B-Chat          | 72 GB         | 4 x RTX 4090 <br> A800 (80GB)               |
| Yi-34B-Chat-4bits    | 20 GB         | RTX 3090  <br> RTX 4090 <br> A10 <br> A30 <br> A100 (40GB) |
| Yi-34B-Chat-8bits    | 38 GB         | 2 x RTX 3090  <br> 2 x RTX 4090 <br> A800  (40GB) |

Below are detailed minimum VRAM requirements under different batch use cases.

|  Model                  | batch=1 | batch=4 | batch=16 | batch=32 |
| ----------------------- | ------- | ------- | -------- | -------- |
| Yi-6B-Chat              | 12 GB   | 13 GB   | 15 GB    | 18 GB    |
| Yi-6B-Chat-4bits  | 4 GB    | 5 GB    | 7 GB     | 10 GB    |
| Yi-6B-Chat-8bits  | 7 GB    | 8 GB    | 10 GB    | 14 GB    |
| Yi-34B-Chat       | 65 GB   | 68 GB   | 76 GB    | > 80 GB   |
| Yi-34B-Chat-4bits | 19 GB   | 20 GB   | 30 GB    | 40 GB    |
| Yi-34B-Chat-8bits | 35 GB   | 37 GB   | 46 GB    | 58 GB    |

## Base models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|----------------------|--------------|:-------------------------------------:|
| Yi-6B                | 15 GB         | RTX3090 <br> RTX4090 <br> A10 <br> A30               |
| Yi-6B-200K           | 50 GB         | A800 (80 GB)                            |
| Yi-34B               | 72 GB         | 4 x RTX 4090 <br> A800 (80 GB)               |
| Yi-34B-200K          | 200 GB        | 4 x A800 (80 GB)                        |