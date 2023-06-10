### Implement some self-supervised learning frameworks on small datasets that are easy to reuse.

- Minimum dependency.
- Easy to reproduce.

### Todos
|        | Cifar10 | Cifar100 | STL10 |
|--------|:---------:|:---------:|:---------:|
| SimCLR | :heavy_check_mark: | :heavy_check_mark: |       |
| BYOL   |         |          |       |
| MoCo   |         |          |       |

### Resnet18 backbone (Top1 accuracy)

|        | Cifar10 | Cifar100 | STL10 |
|--------|:---------:|:----------:|:----------:|
| SimCLR | [91.76](https://tensorboard.dev/experiment/xX41MXS7QqWVB1E1mqUgGw/#scalars&_smoothingWeight=0) |   [65.86](https://tensorboard.dev/experiment/kND6mvhWSDKvgFeKEg5T3Q/#scalars&_smoothingWeight=0)       |       |
| BYOL   |         |          |       |
| MoCo   |         |          |       |

### Resnet50 backbone (Top1 accuracy)

|        | Cifar10 | Cifar100 | STL10 |
|--------|:---------:|:---------:|:---------:|
| SimCLR | [93.23](https://tensorboard.dev/experiment/nqCAT0f8Tdin7lpW6BpcLw/#scalars&_smoothingWeight=0) | [67.43](https://tensorboard.dev/experiment/M7iBEIV1R1OIwuqxjSLJJQ/#scalars&_smoothingWeight=0) |       |
| BYOL   |         |          |       |
| MoCo   |         |          |       |
