# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import argparse
import json
import logging
import subprocess, tempfile

import torch
import torch.distributed as dist
from monai.utils import RankFilter

# Try to import yaml, make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def setup_logging(logger_name: str = "") -> logging.Logger:
    """
    Setup the logging configuration.

    Args:
        logger_name (str): logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    if dist.is_initialized():
        logger.addFilter(RankFilter())
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def load_config_file(config_path: str) -> dict:
    """
    Load configuration from JSON or YAML file based on file extension.
    
    Args:
        config_path (str): Path to configuration file (.json, .yaml, or .yml)
    
    Returns:
        dict: Loaded configuration dictionary
    
    Raises:
        ValueError: If file extension is not supported or YAML is not installed
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    if file_ext == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif file_ext in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ValueError(
                f"YAML file detected ({config_path}) but PyYAML is not installed. "
                "Install it with: pip install pyyaml"
            )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension: {file_ext}. "
            "Supported extensions: .json, .yaml, .yml"
        )


def load_config(env_config_path: str, model_config_path: str, model_def_path: str) -> argparse.Namespace:
    """
    Load configuration from JSON or YAML files.
    
    Automatically detects file format based on extension (.json, .yaml, .yml).

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.

    Returns:
        argparse.Namespace: Loaded configuration.
    """
    args = argparse.Namespace()

    # Load environment config
    env_config = load_config_file(env_config_path)
    for k, v in env_config.items():
        setattr(args, k, v)

    # Load model config
    model_config = load_config_file(model_config_path)
    for k, v in model_config.items():
        setattr(args, k, v)
    
    # Load model definition
    model_def = load_config_file(model_def_path)
    for k, v in model_def.items():
        setattr(args, k, v)

    return args


def initialize_distributed() -> tuple:
    """
    Initialize distributed training. Auto-detects if running under torchrun/DDP
    by checking for RANK and WORLD_SIZE environment variables.
    
    Works for both single-node and multi-node setups.

    Returns:
        tuple: local_rank, world_size, and device.
    """
    # Auto-detect if running under torchrun or other distributed launcher
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Multi-GPU distributed training
        dist.init_process_group(backend="nccl", init_method="env://")
        
        # Get global rank and world size
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Get local rank (GPU ID on current node)
        # LOCAL_RANK is set by torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", global_rank))
        
        # Set the device for this process
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        
        return local_rank, world_size, device
    else:
        # Single GPU or CPU mode (not using torchrun)
        local_rank = 0
        world_size = 1
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return local_rank, world_size, device

def run_torchrun(module, module_args, num_gpus=1):
    num_nodes = 1

    # temp JSON path for outputs
    with tempfile.TemporaryDirectory() as tmpd:
        out_index = os.path.join(tmpd, "outputs.json")
        full_args = module_args + ["--out_index", out_index]

        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--nnodes", str(num_nodes),
            "-m", module,
        ] + full_args

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env
        )

        try:
            # stream stdout
            for line in iter(proc.stdout.readline, ""):
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line.rstrip())
        finally:
            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="")

        # collect result
        if os.path.exists(out_index):
            with open(out_index) as f:
                return json.load(f)  # list of per-rank paths
        return None