import os
import sys
import tqdm
import logging
from transformers import ElectraTokenizer, ElectraForQuestionAnswering

TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer
}

MODEL_FOR_QUESTION_ANSWERING = {
    "koelectra-base-v3": ElectraForQuestionAnswering
}


def init_logger(output_dir) -> None:
    """
    Logging 시작을 위한 config를 정의합니다.
    """
    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            TqdmLoggingHandler()
        ]
    )


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)