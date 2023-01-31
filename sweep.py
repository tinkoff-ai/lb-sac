# inital idea from:
# https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool
import pyrallis
import shlex
import glob
import itertools

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, current_process, Queue
from subprocess import Popen, TimeoutExpired


@dataclass
class Config:
    # main command to run configs with
    command: str = "python train.py"
    # path to the configs to run, can use glob syntax
    configs: str = "configs/*.yaml"
    # sweep configuration, generates commands by grid search
    sweep_config: Optional[Dict[str, List[Any]]] = None
    # number of seeds to run each config
    num_seeds: int = 4
    # number of available gpus to use simultaneously
    num_gpus: int = 4
    # number of processes to run on each gpu simultaneously
    num_proc_per_gpu: int = 1
    # timeout for the runs, 10 days by default
    time_limit: int = 864000
    # will just print all planned commands to run, then exit
    dry_run: bool = False


def init_worker(shared_queue: Queue):
    global queue
    queue = shared_queue


# https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
def __run_process(command: str, timeout: float):
    print(f"Running command: {command}")
    process = Popen(shlex.split(command))
    try:
        return_code = process.wait(timeout=timeout)
        assert return_code == 0
    except TimeoutExpired:
        print("TIME OUT!!!", process.pid)
        process.terminate()


def run_process(base_command: str, timeout: float):
    global queue

    gpu_id = queue.get()
    try:
        ident = current_process().ident
        print('{}: starting process on GPU {}'.format(ident, gpu_id))
        command = f"{base_command} --device cuda:{gpu_id}"
        __run_process(command, timeout=timeout)
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)


def generate_sweep_commands(
        commands: List[str], sweep_config: Dict[str, List[Any]]
) -> List[str]:
    # simple grid search generation for now
    sweep_grid = itertools.product(*list(sweep_config.values()))
    sweep_keys = list(sweep_config.keys())

    new_commands = []
    for combination in sweep_grid:
        suffix = [f"--{key} {str(value)}" for key, value in zip(sweep_keys, combination)]
        for command in commands:
            new_command = " ".join([command, *suffix])
            new_commands.append(new_command)

    return new_commands


def download_all_datasets():
    import gym
    import d4rl

    for name in [
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2"
        "halfcheetah-random-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-full-replay-v2",
        "walker2d-random-v2",
        "walker2d-expert-v2",
        "walker2d-full-replay-v2",
        "hopper-random-v2",
        "hopper-expert-v2",
        "hopper-full-replay-v2",
        "ant-medium-v2"
        ]:
        env = gym.make(name).get_dataset()


if __name__ == '__main__':
    # generate all commands to run
    config = pyrallis.parse(config_class=Config)

    commands = []
    for config_path in glob.glob(config.configs):
        commands.append(f"{config.command} --config_path {config_path}")

    sweep_config = {
        "train_seed": list(range(0, config.num_seeds))
    }
    if config.sweep_config is not None:
        sweep_config.update(config.sweep_config)

    commands = generate_sweep_commands(commands, sweep_config=sweep_config)
    # run all commands on all available gpus
    if config.dry_run:
        print(f"All commands to be executed ({len(commands)} in total):")
        print(*commands, sep="\n")
    else:
        # pre-download datasets if needed
        # download_all_datasets()

        shared_queue = Queue()
        # initialize the queue with the GPU ids
        for gpu_id in range(config.num_gpus):
            for _ in range(config.num_proc_per_gpu):
                shared_queue.put(gpu_id)

        pool = Pool(processes=config.num_gpus * config.num_proc_per_gpu, initializer=init_worker, initargs=(shared_queue,))
        pool.starmap(run_process, zip(commands, itertools.repeat(config.time_limit)))
        pool.close()
        pool.join()