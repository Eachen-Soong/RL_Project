## RL Final Project1: Starter Code

We provide the starter code based on PyTorch for the interaction with different gym environments. 

You are encouraged to use the code style provided but feel free to complete the entire code by yourself.

### Getting started

To start with, you can run the following command to create an anaconda environment and install required dependencies.

```
conda env create -f conda_env.yal
```

After this installation you can activate your environment with:

```
source activate env_name
```

### Instructions

To train an agent for specific target task, you may enter the target task folder which has `train.py`, and run the following command.

```
python train.py \
	--task target_task \
	--train_eps 100
```

You may also specify the seed value by adding `--seed`and `--random_seed` flags after the command above. You may customize other flags for convenience as well.

### Results

You are encouraged to use Tensorboard to trace your training results, with which you should find logs in the working directory you specified. You can monitor your training by entering the target working directory and run the following command.

```
tensorboard --logdir .
```

You will be able to see plots by opening up tensorboard in your browser.

Or instead, you can simply output the results in your console, and maintain a buffer to record the variables during training process. Either way, plots of training variables will be great help for your analysis on your code.