# Moon Landing Deep Q Learning

## Introduction
A Reinforcement Learning AI Agent that use Deep Q Network to play Lunar Lander. The code is implemented with `Tensorflow` and `PyTorch`.

## Hyper Parameters
* Algorithm: Deep Q-Network with a Double Fully connected layers
* Each Neural Network has the same structure: 2 Fully connected layers each with 128 nodes.
* Optimization algorithm: Adaptive Moment (Adam)
* Learning rate: α = 0.0001
* Discount factor: γ = 0.99
* Minimum exploration rate: ε = 0.1
* Replay memory size: 10^6
* Mini batch size: 2^6

## Problem Description
* The agent has to learn how to land a Lunar Lander to the moon surface safely, quickly and accurately.
* If the agent just lets the lander fall freely, it is dangerous and thus get a very negative reward from the environment.
* If the agent does not land quickly enough (after 20 seconds), it fails its objective and receive a negative reward from the environment.
* If the agent lands the lander safely but in wrong position, it is given either a small negative or small positive reward, depending on how far from the landing zone is the lander.
* If the AI lands the lander to the landing zone quickly and safely, it is successful and is award very positive reward.

## Q Learning
Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

### Formula
![image](https://user-images.githubusercontent.com/50926437/160830926-facf7d08-bbba-4f2b-ab1b-ccacf8ec2010.png)

### Agent and Environment interaction
![image](https://user-images.githubusercontent.com/50926437/160830697-267cd582-b902-4110-b47f-8c3c6df6eec7.png)
![image](https://user-images.githubusercontent.com/50926437/160830791-671d4989-7b1b-4f45-a721-32d4d54571c6.png)

## Result

### Before Training
![image](https://user-images.githubusercontent.com/50926437/160831105-0fccc137-806b-47a1-8818-d860e32fe9a3.png)

### After Training
![image](https://user-images.githubusercontent.com/50926437/160831135-edce5b06-63a1-4218-9b55-1e78d91362e4.png)

### Learning Curve
![image](https://user-images.githubusercontent.com/50926437/160831218-5a013b66-d189-4890-ba51-cecd941caebc.png)

* The Blue curve shows the reward the agent earned in each episode.
* The Red curve shows the average reward from the corresponding episode in the x-axis and 100 previous episodes. In other words, it shows the average reward of 100 most current episodes.
* From the plot, we see that the Blue curve is much noisier due to exploration ε = 0.1 throughout the training process and due to the imperfect approximation during some first episodes of the training.
* Averaging 100 most current rewards produces much smoother curve, however.
* From the curve, we can conclude that the agent has successfully learned a good policy to solve the Lunar Lander problem, according to OpenAI criteria (the average point of any 100 consecutive episodes is at least 200).

*Made By Amirhossein Abaskohi*
