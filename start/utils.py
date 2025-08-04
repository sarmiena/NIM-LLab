# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
from pathlib import Path
from typing import List, Optional

from nim_llm_sdk.engine import NimAsyncEngineArgs
from nim_llm_sdk.hub import TLLM_CONFIG_FILE_ARG_MAPPING
from nim_llm_sdk.hub.repo import NimRepo
from nim_llm_sdk.logger import init_logger

logger = init_logger(__name__)


# Logs an error message and exits with code 1
def error_and_exit(error_message: str):
    logger.error(error_message)
    sys.exit(1)


def list_files_recursive(path: Path, extension: str = "*.json"):
    if not path.is_dir():
        raise NotADirectoryError(f"'{str(path)}' is not a directory.")

    return [ConfigPath(local_path=str(file_path)) for file_path in path.rglob(extension) if file_path.is_file()]


class ConfigPath:
    # Stores configuration file paths either from NIM cache or local disk.
    # `repo_path` points to files stored in NIM cache
    # `local_path` points to files stored in user provided local model paths
    def __init__(self, local_path: str = None, repo_path: str = None):
        self.local_path = local_path
        self.repo_path = repo_path

    @property
    def from_local(self):
        return self.local_path is not None

    @property
    def from_repo(self):
        return self.repo_path is not None

    def get_path(self):
        if self.from_local:
            return self.local_path
        elif self.from_repo:
            return self.repo_path
        else:
            raise ValueError("Neither repo nor local paths are set. Invalid `ConfigPath` object.")

    def __len__(self):
        return len(self.get_path())


def get_all_config_paths_from_repo_and_local(local_path: Path, repo: Optional[NimRepo] = None) -> List[ConfigPath]:
    """
    Recursively finds all .json files from local model path and manifest repository
    """
    all_files_list = set(list_files_recursive(local_path))
    if repo is not None:
        repo_files_list = [ConfigPath(repo_path=file) for file in repo.files() if file.endswith(".json")]
        all_files_list = all_files_list.union(repo_files_list)

    all_files_list = list(all_files_list)
    all_files_list.sort(key=lambda s: len(s))
    # order matters here
    # attempts to set configs from root folder to sub-folders
    logger.debug(
        f"Found following config paths in local path: {local_path} and current profile: {[file.get_path() for file in all_files_list]}."
    )
    return all_files_list


def set_all_config_paths_in_engine_args(
    engine_args: NimAsyncEngineArgs, config_paths: List[ConfigPath], repo: Optional[NimRepo] = None
):
    """
    Sets TRTLLM engine config, pre-trained config and runtime config json strings from the List of configuration paths found using `get_all_config_paths_from_repo_and_local()`
    """
    for file in TLLM_CONFIG_FILE_ARG_MAPPING:
        for config_path in config_paths:
            if config_path.get_path().endswith(file):
                if config_path.from_repo:
                    if file in repo.files():
                        config_path = repo.get(file).path()
                    else:
                        continue
                elif config_path.from_local:
                    config_path = config_path.local_path
                else:
                    continue
                with open(config_path, 'r') as fh:
                    # dont set engine args if already set
                    # sorted order of config paths ensures a specifc priority rank
                    if getattr(engine_args, TLLM_CONFIG_FILE_ARG_MAPPING[file]) is None:
                        setattr(engine_args, TLLM_CONFIG_FILE_ARG_MAPPING[file], fh.read())
                        logger.debug(f"Loading `{TLLM_CONFIG_FILE_ARG_MAPPING[file]}` from: {config_path}.")
                        break


def get_path_to_gguf_model(model_path: str) -> str:
    gguf_files = []
    for filepath in list_files_recursive(Path(model_path), "*.gguf"):
        gguf_files.append(filepath.local_path)
    if not len(gguf_files):
        raise ValueError(f"{model_path} does not have any GGUF files. Unknown input format.")

    gguf_files.sort()

    if len(gguf_files) > 1:
        logger.info(f"Found following GGUF file(s) in {model_path}: {gguf_files}. NIM will serve: {gguf_files[0]}.")
    # use the first gguf file found
    return gguf_files[0]


def log_folder_structure(root_dir, logger=None, prefix=""):
    entries = sorted(os.listdir(root_dir))
    logger.info(f"Found following files in {root_dir}")
    for idx, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "└── " if idx == len(entries) - 1 else "├── "
        logger.info(f"{prefix}{connector}{entry}")
        if os.path.isdir(path):
            extension = "    " if idx == len(entries) - 1 else "│   "
            log_folder_structure(path, logger, prefix + extension)


def log_expected_folder_structure(logger):
    logger.info(
        """\
Checkpoint directory structure expectations for different model formats:

1. HuggingFace safetensors

├── config.json                           # [Required] HuggingFace model confifuration 
├── model-00001-of-00004.safetensors      # [Required] Model weights stored as safetensors 
├── model-00002-of-00004.safetensors 
├── model-00003-of-00004.safetensors 
├── model-00004-of-00004.safetensors 
├── ...
├── generation_config.json                # [Optional] Parameters to guide text generation
├── model.safetensors.index.json          # [Optional] Weights mapping 
├── special_tokens_map.json               # [Optional] Special tokens mapping 
├── tokenizer.json                        # [Optional] Tokenization method, vocabulary, pre-tokenization rules etc 
└── tokenizer_config.json                 # [Optional] Configuration details for a specific model's tokenizer

2. Unified HuggingFace safetensors

├── config.json                           # [Required] HuggingFace model confifuration 
└── hf_quant_config.json                  # [Required] HuggingFace quantization configuration
├── model-00001-of-00004.safetensors      # [Required] Model weights stored as safetensors 
├── model-00002-of-00004.safetensors 
├── model-00003-of-00004.safetensors 
├── model-00004-of-00004.safetensors 
├── ...
├── generation_config.json                # [Optional] Parameters to guide text generation
├── model.safetensors.index.json          # [Optional] Weights mapping 
├── special_tokens_map.json               # [Optional] Special tokens mapping 
├── tokenizer.json                        # [Optional] Tokenization method, vocabulary, pre-tokenization rules etc 
├── tokenizer_config.json                 # [Optional] Configuration details for a specific model's tokenizer


3. HuggingFace GGUF 

├── config.json                             # [Required] HuggingFace model confifuration 
├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  # [Required] GGUF model weights stored as safetensors 
├── ...
├── generation_config.json                  # [Optional] Parameters to guide text generation
├── model.safetensors.index.json            # [Optional] Weights mapping 
├── special_tokens_map.json                 # [Optional] Special tokens mapping 
├── tokenizer.json                          # [Optional] Tokenization method, vocabulary, pre-tokenization rules etc 
└── tokenizer_config.json                   # [Optional] Configuration details for a specific model's tokenizer

4. TRTLLM checkpoints

├── config.json                           # [Required] HuggingFace model confifuration 
├── generation_config.json                # [Optional] Parameters to guide text generation
├── model.safetensors.index.json          # [Optional] Weights mapping 
├── special_tokens_map.json               # [Optional] Special tokens mapping 
├── tokenizer.json                        # [Optional] Tokenization method, vocabulary, pre-tokenization rules etc 
├── tokenizer_config.json                 # [Optional] Configuration details for a specific model's tokenizer
└── trtllm_ckpt
    ├── config.json                       # [Required] TRTLLM pretrained configuration
    ├── rank0.safetensors                 # [Required] TRTLLM checkpoint safetensors
    ├── ...

5. TRTLLM engines

├── config.json                           # [Required] HuggingFace model confifuration 
├── generation_config.json                # [Optional] Parameters to guide text generation
├── model.safetensors.index.json          # [Optional] Weights mapping 
├── special_tokens_map.json               # [Optional] Special tokens mapping 
├── tokenizer.json                        # [Optional] Tokenization method, vocabulary, pre-tokenization rules etc 
├── tokenizer_config.json                 # [Optional] Configuration details for a specific model's tokenizer
└── trtllm_engine
    ├── config.json                       # [Required] TRTLLM engine configuration
    ├── rank0.engine                      # [Required] TRTLLM serialized engine
    ├── ...

"""
    )
