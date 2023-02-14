import os
import json
from glob import glob
from datetime import datetime
from argparse import ArgumentParser

import torch
import datasets
import transformers
from evaluate import load
from attrdict import AttrDict
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForQuestionAnswering, EvalPrediction, TrainingArguments, DataCollatorWithPadding

from data import preprocess_training_examples, preprocess_validation_examples, post_processing_function
from trainer_qa import QuestionAnsweringTrainer
from modeling import RobertaBiGRUForQuestionAnswering

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

def main(cli_args):
    # [학습을 위한 Arguments 준비]
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))

    # [SEED 고정]
    set_seed(args.seed)

    # [CheckPoint 경로 설정]
    now = datetime.now().strftime('%y%m%d%H%M')  # 연월일시분
    under_ckpt_dir = f"{cli_args.config_file.split('.')[0]}_{now}"
    args.ckpt_dir = os.path.join(args.ckpt_dir, under_ckpt_dir)

    # [Model & Tokenizer 불러오기]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaBiGRUForQuestionAnswering.from_pretrained(args.model_name_or_path)

    # [Dataset 불러오기]
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files['train'] = glob(os.path.join(args.train_file, '**'))
        if args.validation_file is not None:
            data_files['validation'] = glob(os.path.join(args.validation_file, '**'))
        if args.test_file is not None:
            data_files['test'] = glob(os.path.join(args.test_file, '**'))
        extension = data_files['train'][0].split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')

    # [Dataset 전처리]
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length, "stride": args.doc_stride}
    )
    eval_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length, "stride": args.doc_stride}
    )
    if args.do_eval:
        test_dataset = raw_datasets["test"].map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length, "stride": args.doc_stride}
        )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    total_batch_size = args.per_device_train_batch_size * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * args.num_train_epochs)
    n_warmup_steps = int(n_total_iterations * args.warmup_ratio)

    metric = load("squad")
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=n_warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        report_to=['wandb'],
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        fp16=True,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        run_name=args.run_name
    )

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=raw_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # 모델 학습
    trainer.train()
    trainer.save_model()

    # 모델 평가
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == '__main__':
    cli_parser = ArgumentParser()
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default = 'klue_roberta_base.json')
    main(cli_parser.parse_args())
