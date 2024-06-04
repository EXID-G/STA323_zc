import os
import pickle
import numpy as np
import torch
import shutil
import tempfile
import argparse
from ray.air import session
from ray.tune import CLIReporter
from datasets import load_dataset, Dataset, load_metric
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.train.huggingface.transformers

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAY_memory_usage_threshold"] = "0.8"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

# 预处理函数
def preprocess_function(examples, mytokenizer):
    inputs = [str(ex) for ex in examples['input']]
    targets = [str(ex) for ex in examples['output']]
    model_inputs = mytokenizer(inputs, max_length=512, truncation=True)
    labels = mytokenizer(targets, max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 保存数据集
def save_dataset(dataset, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

# 加载预处理数据集
def load_preprocessed_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 加载和预处理数据集
def load_and_preprocess_datasets(tokenizer, train_file, validation_file, test_file):
    train_save_path = 'data/mydata/preprocessed/train.pkl'
    val_save_path = 'data/mydata/preprocessed/val.pkl'
    test_save_path = 'data/mydata/preprocessed/test.pkl'

    if os.path.exists(train_save_path) and os.path.exists(val_save_path) and os.path.exists(test_save_path):
        train_dataset = load_preprocessed_dataset(train_save_path)
        val_dataset = load_preprocessed_dataset(val_save_path)
        test_dataset = load_preprocessed_dataset(test_save_path)
    else:
        train_dataset = load_dataset('csv', data_files=train_file, cache_dir='data/cachefile')['train']
        val_dataset = load_dataset('csv', data_files=validation_file, cache_dir='data/cachefile')['train']
        test_dataset = load_dataset('csv', data_files=test_file, cache_dir='data/cachefile')['train']

        train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
        test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

        save_dataset(train_dataset, train_save_path)
        save_dataset(val_dataset, val_save_path)
        save_dataset(test_dataset, test_save_path)

    return train_dataset, val_dataset, test_dataset

# 训练函数
def train_func(config, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset, val_dataset, _ = load_and_preprocess_datasets(tokenizer, args.train_file, args.validation_file, args.test_file)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1).flatten()
        labels = labels.flatten()
        metric = load_metric("accuracy")
        return metric.compute(predictions=predictions, references=labels)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ray.train.huggingface.transformers.RayTrainReportCallback()],
    )

    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()

    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss")
    if val_loss is None:
        raise ValueError("Evaluation results did not contain 'eval_loss'")

    metrics = {"eval_loss": val_loss, "epoch": config["num_epochs"]}
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        trainer.save_model(temp_checkpoint_dir)
        trainer.save_state()
        trainer.save_metrics("eval", eval_results)
        tokenizer.save_pretrained(temp_checkpoint_dir)
        session.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir))

    if ray.train.get_context().get_world_rank() == 0:
        print(metrics)

# 调优函数
def tune_transformer(args):
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8, 16, 32]),
        "num_epochs": tune.choice([1]),
        "weight_decay": tune.uniform(0.0, 0.3),
    }

    scheduler = ASHAScheduler(
        metric="eval_accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "num_train_epochs", "weight_decay"],
        metric_columns=["eval_accuracy", "eval_loss", "epoch", "training_iteration"],
        max_report_frequency=10,
        print_intermediate_tables=False
    )
    analysis = tune.run(
        tune.with_parameters(train_func, args=args),
        resources_per_trial={"cpu": 0.2},
        config=search_space,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_qa_model",
        local_dir=args.output_dir,
        stop={"training_iteration": 2},
        keep_checkpoints_num=3,
        checkpoint_score_attr="eval_acc",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    best_trial = analysis.get_best_trial(metric="eval_loss", mode="min")
    best_model_path = best_trial.checkpoint.value
    destination_path = os.path.join(args.output_dir, "best_model")
    shutil.copytree(best_model_path, destination_path)

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--train_file", type=str, required=True, help="The input training data file (a csv or json file).")
    parser.add_argument("--validation_file", type=str, required=True, help="The input validation data file (a csv or json file).")
    parser.add_argument("--test_file", type=str, required=True, help="The input test data file (a csv or json file).")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length", type=int, default=384, help="The maximum total input sequence length after tokenization.")
    # parser.add_argument("--doc_stride", type=int, default=128, help="The stride to take between chunks when splitting up a long document.")

    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    tune_transformer(args)

    #? python Q1_2.py  --model_name_or_path /STA323_zc/proj2/flan-t5-small --train_file /STA323_zc/proj2/data/mydata/mytrain.csv  --validation_file /STA323_zc/proj2/data/mydata/myvalidation.csv --test_file /STA323_zc/proj2/data/mydata/mytest.csv --output_dir /STA323_zc/proj2/ray_results