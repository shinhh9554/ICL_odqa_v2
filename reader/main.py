import os
import json
from argparse import ArgumentParser

from attrdict import AttrDict
import torch.distributed as dist


def ddp_setup():
    """
    torch.distributed 초기화 수행 함수
    """
    dist.init_process_group(backend='nccl')


def parse_args():
    """
    모델 학습을 위한 Arguments를 준비하는 함수

    Returns:
        args (AttrDict): 모델 학습을 위한 Arguments
    """
    cli_parser = ArgumentParser()
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default='koelectra-base-v3.json')
    cli_args = cli_parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))

    return args


def main():
    # [모델 학습을 위한 Arguments 준비]
    args = parse_args()

    # [torch.distributed 초기화]
    ddp_setup()
    



if __name__ == "__main__":
    main()


    main(cli_args)
