import numpy as np
from collections import deque
from gym import wrappers
import helper
import os


def _run_train_episode(agent, env, config, epsilon):
    """Run one full training episode and return the achieved score"""
    state = env.reset()
    score = 0
    for t in range(config["train"]["episode_length"]):
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.save_experiences(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        score += reward
        if done:
            break
    return score


def train(agent, env, config):
    """Deep Q-Learning session"""

    scores = []
    scores_window = deque(maxlen=100)
    statistics = {"mean": [], "std": []}
    epsilon = config["train"]["epsilon_high"]
    for i_episode in range(1, config["train"]["nb_episodes"] + 1):
        score = _run_train_episode(agent, env, config, epsilon)
        scores.append(score)
        scores_window.append(score)
        statistics["mean"].append(np.mean(scores_window))
        statistics["std"].append(np.std(scores_window))

        epsilon = max(config["train"]["epsilon_low"], config["train"]["epsilon_decay"] * epsilon)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= config["general"]["average_score_for_solving"]:
            print("\n{} Environment solved in {:d} episodes!".format(config["general"]["env_name"], i_episode - 100))
            break
    best_mean, best_std = max(zip(statistics["mean"], statistics["std"]))
    return scores, best_mean, best_std


def _set_up_monitoring(env, config):
    """wrap the environment to allow rendering and set up a save directory"""
    helper.mkdir(os.path.join(".",
                              *config["general"]["monitor_dir"],
                              config["general"]["env_name"]))
    current_date_time = helper.get_current_date_time()
    current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")

    env = wrappers.Monitor(env, os.path.join(".",
                                             *config["general"]["monitor_dir"],
                                             config["general"]["env_name"],
                                             current_date_time))
    return env


def test(agent, env, config):
    """run and render one episode and save a video file"""
    env = _set_up_monitoring(env, config)

    state = env.reset()
    score = 0
    done = False
    while done is False:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        score += reward
    return score



