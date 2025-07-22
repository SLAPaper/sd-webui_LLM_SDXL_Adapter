# Copyright 2025 SLAPaper
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib


def preload(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-sdxl-adapter-gemma-path",
        type=pathlib.Path,
        help="gemma3-1b path for SDXL LLM adapter",
    )
    parser.add_argument(
        "--llm-sdxl-adapter-model-path",
        type=pathlib.Path,
        help="model path for SDXL LLM adapter",
    )
