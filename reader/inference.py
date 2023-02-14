import os
from glob import glob
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding

from data import preprocess_validation_examples, post_processing_function

def parse_args():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = ArgumentParser()

    p.add_argument('--model_path', default="ckpt/mrc_2302130040")
    p.add_argument('--dataset_name', default=None)
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument('--test_file', default="data/test")
    p.add_argument("--max_length", default=512)
    p.add_argument("--stride", default=128)
    p.add_argument('--batch_size', type=int, default=500)
    p.add_argument('--do_eval', type=bool, default=False)
    p.add_argument('--do_predict', type=bool, default=True)

    config = p.parse_args()

    return config
def inference():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)

    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.test_file is not None:
            data_files['test'] = glob(os.path.join(args.test_file, '**'))
        extension = data_files['test'][0].split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')

    test_dataset = raw_datasets["test"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "stride": args.stride}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    with torch.no_grad():
        inference_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
        test_dataloader = DataLoader(
            inference_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size, shuffle=False
        )

        # Don't forget turn-on evaluation mode.
        model.eval()

        # Predictions
        start_logits = []
        end_logits = []
        for batch in tqdm(test_dataloader):
            batch = batch.to(device)
            outputs = model(**batch)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(inference_dataset_for_model)]
        end_logits = end_logits[: len(inference_dataset_for_model)]

    metric = load("squad")
    def compute_metrics(p):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    eval_result = post_processing_function(raw_datasets["test"], test_dataset, (start_logits, end_logits), args)
    if args.do_eval:
        result = compute_metrics(eval_result)
        print(result)
    elif args.do_predict:
        questions = raw_datasets["test"]["question"]
        contexts = raw_datasets["test"]["context"]
        for answer in eval_result:
            ex_id = raw_datasets["test"]["id"].index(answer["id"])
            print(questions[ex_id])
            print(contexts[ex_id])
            print(answer)
            print()

if __name__ == '__main__':
    inference()