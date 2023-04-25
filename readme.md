# Dataset Distillation with Convexified Implicit Gradients
Code for the paper ["Dataset Distillation with Convexified Implicit Gradients"](https://arxiv.org/abs/2302.06755)

Contact: [Noel Loo](loo@mit.edu)

# Abstract
We propose a new dataset distillation algorithm using reparameterization and convexification of implicit gradients (RCIG), that substantially improves the state-of-the-art. To this end, we first formulate dataset distillation as a bi-level optimization problem. Then, we show how implicit gradients can be effectively used to compute meta-gradient updates. We further equip the algorithm with a convexified approximation that corresponds to learning on top of a frozen finite-width neural tangent kernel. Finally, we improve bias in implicit gradients by parameterizing the neural network to enable analytical computation of final-layer parameters given the body parameters. RCIG establishes the new state-of-the-art on a diverse series of dataset distillation tasks. Notably, with one image per class, on resized ImageNet, RCIG sees on average a 108% improvement over the previous state-of-the-art distillation algorithm. Similarly, we observed a 66% gain over SOTA on Tiny-ImageNet and 37% on CIFAR-100.

# Example usage
To distill on CIFAR-10 with 10 images/cls:
`python3 distill_dataset.py --dataset_name cifar10 --n_images 10 --output_dir ./output_dir/ --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0`

To eval on CIFAR-10 with 10 images/cls (using pre-distilled datasets provided):
`python3 eval.py --dataset_name cifar10 --checkpoint_path ./distilled_images_final/0/cifar10_10/checkpoint_10000 --config_path ./configs_final/depth_3.txt --random_seed 0`

We include pre-distilled datasets for CIFAR-10 with 10 img/cls, CIFAR-10 1/cls, CIFAR-100 1/cls and Tiny-Imagenet 1/cls.

We recommend using the configs located in `./configs_final/` when distilling datasets as these were what we used to create the results in the paper. These contain all the hyperparameters used for distillation. For a description of these hyperparameters, see `./configs_final/example_config.txt`

Specifically the configs we use are:

- For MNIST, and Fashion-MNIST, we use the config `depth_3_no_flip.txt`
- For CIFAR-10, CIFAR-100, and CUB-200, we use config `depth_3.txt`
- For CIFAR-100 with 50 ipc we use config `cifar100_50.txt`
- For Tiny-ImageNet 1 ipc, we use config `depth_4_200.txt`
- For Tiny-ImageNet 10 ipc and resized Imagenet, we use config `depth_4_big.txt`
- For ImageNette and ImageWoof, we use config `depth_5.txt`

For the number of training iterations (i.e. the --max_steps argument, see section S.4.2 in the appendix)

# Citation
Please cite our paper as
```
@misc{https://doi.org/10.48550/arxiv.2302.06755,
  doi = {10.48550/ARXIV.2302.06755},
  url = {https://arxiv.org/abs/2302.06755},
  author = {Loo, Noel and Hasani, Ramin and Lechner, Mathias and Rus, Daniela},
  title = {Dataset Distillation with Convexified Implicit Gradients},
  publisher = {arXiv},
  year = {2023},
}```
