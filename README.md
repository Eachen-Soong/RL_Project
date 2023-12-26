## RL Final Project

We tested different algorithms on different gym environments (cliff walking, pendulum, and mountain car continuous).

### Getting started

To start with, you can run the following command to create an anaconda environment and install required dependencies.

```
pip install -r requirements.txt
```

### Instructions

In Linux system, you can use the scripts in `./scripts` to search adquate hyperparameters. Example:
```
bash scripts/MountainCarDDPGSearch.sh
```
To train an agent for specific target task, you may enter the target task folder, and run the following command. Example:

```
python Pendulum/DDPG.py --max_episode=10000 --tau=0.005 --exploration_noise=0.1
```

You may also specify the seed value by adding `--seed`and `--random_seed` flags after the command above. You may customize other flags for convenience as well.

### Test your models
After trained a model, you can test it. Example:
```
python Pendulum/DDPG.py --mode=test --record_test=True --tau=0.005 --exploration_noise=0.1
```