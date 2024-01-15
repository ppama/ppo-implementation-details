import argparse
import os
import random
import time
from distutils.util import strtobool

#import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`") # useful for reproducing experiments
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, # uses default entity, username
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel game environments')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    #print(args) #useful when changing config, do it in cli instead of modifying code 
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb


        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # set up tensorboard, visualization of training losses or episodic returns
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    #for i in range(100): # test
    #    writer.add_scalar("test_loss", i*2, global_step=i)
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # set up vector environment

    env = gym.make("CartPole-v1") # initialize env
    env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
    env = gym.wrappers.RecordVideo(env, "videos", record_video_trigger=lambda t : t % 100 == 0) # records a video every 100 episodes
    observation = env.reset() # reset env, get first obs
    for _ in range(200):
        action = env.action_space.sample() # sample an action to step env
        observation, reward, terminated, truncated, info = env.step(action) 
        if terminated or truncated:
            observation = env.reset()
            print(f"episodic return: {info['episode']['r']}") # wrapper populates info variable with episodic return
        env.close()

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk(): # function that makes a gym environment
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0: # only record video for the first worker(sub-enviroment)
                    env = gym.wrappers.RecordVideo(env, "videos", record_video_trigger=lambda t : t % 1000 == 0)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    # API for creating a vector environment by passing a list of env creating functions
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)]) # vectorized environments are synchronized environments that run in parallel
    observation = envs.reset()
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
        for item in info:
            if "episode" in item.keys():
                print(f"episodic return: {item['episode']['r']}")
                # NOTE: there is no `observation = env.reset()` here, this is because the vector environments automatically reset when all envs are terminated
