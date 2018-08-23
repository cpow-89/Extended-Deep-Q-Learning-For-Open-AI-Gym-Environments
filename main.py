import gym
import json
import os
import helper
import sessions
import argparse
from dqn_agent import Agent


def main():
    parser = argparse.ArgumentParser(description="Run Extended Q-Learning with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    args = parser.parse_args()

    # load config files
    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        config = json.load(read_file)

    env = gym.make(config["general"]["env_name"])
    env.seed(config["general"]["seed"])
    agent = Agent(config=config)
    # watch an untrained agent
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

    if config["train"]["run_training"]:
        scores = sessions.train(agent, env, config)
        helper.plot_scores(scores)
        agent.save()
    else:
        agent.load()
        sessions.test(agent, env, config)


if __name__ == "__main__":
    main()
