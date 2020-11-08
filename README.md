# Target-driven 2/3/6 DoF Visual Navigation Model using Deep Reinforcement Learning

This repository tries to extends the work [Target-drive Visual Navigation using Deep Reinforcement Learning](https://www-cs.stanford.edu/groups/vision/pdf/zhu2017icra.pdf) to support RL agent action space of 2/3/6DoF. The code is in PyTorch and the simulator used is [Habitat Sim](https://github.com/facebookresearch/habitat-sim).

Major changes in the repository:
* Added support for Habitat Simulator
* Made changes in configuration to support high degree of freedom (2/3/6DoFs) in action space.

## Introduction

This repository provides a Pytorch implementation of the deep siamese actor-critic model for indoor scene navigation introduced in the following paper:

**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](https://www-cs.stanford.edu/groups/vision/pdf/zhu2017icra.pdf)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)

## Setup and run
This code is implemented in [Pytorch 1.4](https://pytorch.org/) and uses Habitat as the simulator. Follow steps provided at [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for simulator installation.

In order to start training, run those commands:
```
git clone https://github.com/pushkalkatara/visual-navigation-agent-pytorch.git
python train.py
```

## Scenes
We use Arkansaw, Ballou, Hillsdale, Roane, Stokes environments from [Gibson-Habitat](https://github.com/facebookresearch/habitat-sim/blob/master/README.md) dataset to perform the experiment.

## Acknowledgements
I would like to acknowledge the following references that have offered great help for me to implement the model.
* ["Habitat Baselines"](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines)
* ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [David Silver's Deep RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [muupan's async-rl repo](https://github.com/muupan/async-rl/wiki)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)
* [Pytorch A3C implementation repo](https://github.com/ikostrikov/pytorch-a3c)

## License
MIT