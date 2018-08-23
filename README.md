# Extended Deep Q-Learning for Multilayer Perceptron

This project includes the code for an extended version of the Deep Q-Learning algorithm which I wrote during my [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) @ Udacity. 
The code is inspired by the [Dvanilla DQN algorithm](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn) provided by Udacity.

Deep Q-Learning for Multilayer Perceptron<br>
\+ Fixed Q-Targets<br>
\+ Experience Replay<br>
\+ Gradient Clipping<br>
\+ Double Deep Q-Learning<br>
\+ Dueling Networks<br>

For more information on the implemented features refer to Extended_Deep_Q_Learning_for_Multilayer_Perceptron.ipynb. The notebook includes a summary of all essential concepts used in the code. It also contains three examples where the algorithm is used to solve Open AI gym environments.

### Examples


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cpow-89/Extended-Deep-Q-Learning-For-Open-AI-Gym-Environments/master/images/Lunar_Lander_v2.gif "Trained Agents1"

#### Lunar_Lander_v2

![Trained Agents1][image1]

[image2]: https://raw.githubusercontent.com/cpow-89/Extended-Deep-Q-Learning-For-Open-AI-Gym-Environments/master/images/CartPole_v1.gif "Trained Agents2"

#### CartPole_v1

![Trained Agents2][image2]

### Dependencies

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Install OpenAi Gym

> git clone https://github.com/openai/gym.git<br>
> cd gym<br>
> pip install -e .<br>
> pip install -e '.[box2d]'<br>
> pip install -e '.[classic_control]'<br>
> sudo apt-get install ffmpeg<br>

3. Install Sourcecode dependencies

> conda install -c rpi matplotlib <br>
> conda install -c pytorch pytorch <br>
> conda install -c anaconda numpy <br>

### Instructions

You can run the project via Extended_Deep_Q_Learning_for_Multilayer_Perceptron.ipynb or running the main.py file through the console.



open the console and run: python main.py -c "your_config_file".json 
optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "Lunar_Lander_v2".json 

#### Config File Description

**"general"** : <br>
> "env_name" : "LunarLander-v2", # The gym environment name you want to run<br>
> "monitor_dir" : ["monitor"], # monitor file direction<br>
> "checkpoint_path": ["checkpoints"], # checkpoint file direction<br>
> "seed": 0, # random seed for numpy, gym and pytorch<br>
> "state_size" : 8, # number of states<br>
> "action_size" : 4, # number of actions<br>
> "average_score_for_solving" : 200.0 # border value for solving the task<br>

**"train"** : 
> "nb_episodes": 2000, # max number of episodes<br>
> "episode_length": 1000, # max length of one episode<br>
> "batch_size" : 256, # memory batch size<br>
> "epsilon_high": 1.0, # epsilon start point<br>
> "epsilon_low": 0.01, # min epsilon value<br>
> "epsilon_decay": 0.995, # epsilon decay<br>
> "run_training" : true # do you want to train? Otherwise run a test session<br>

**"agent"** :
> "learning_rate": 0.0005, # model learning rate<br>
> "gamma" : 0.99, # reward weight<br>
> "tau" : 0.001, # soft update factor<br>
> "update_rate" : 4 # interval in which a learning step is done<br>

**"buffer"** :
> "size" : 100000 # experience replay buffer size<br>

**"model"** :
> "fc1_nodes" : 256, # number of fc1 output nodes<br>
> "fc2_adv" : 256, # number of fc2_adv output nodes<br>
> "fc2_val" : 128 # number of fc2_val output nodes<br>
