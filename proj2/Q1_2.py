import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["RAY_memory_usage_threshold"] = "0.8"
# os.environ["RAY_memory_monitor_refresh_ms"] = "0"

# 设置环境变量以禁用严格的度量检查
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import pickle
import shutil
import tempfile
import argparse

import numpy as np
import ray
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    T5ForConditionalGeneration,
)

from ray.air import session
import ray.train.huggingface.transformers
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
from ray import tune


def preprocess_function(examples,mytokenizer):
    # Ensure inputs and targets are lists of strings
    inputs = [str(ex) for ex in examples['input']]
    targets = [str(ex) for ex in examples['output']]

    # Tokenize inputs
    model_inputs = mytokenizer(inputs, max_length=384, padding= True, truncation=True,return_tensors="pt")
    # Tokenize targets
    # with tokenizer.as_target_tokenizer():
    labels = mytokenizer(targets, max_length=30,padding = True,truncation=True,return_tensors="pt")
    # Add labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def save_dataset(dataset, path):
    # dataset.save_to_disk(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

def load_preprocessed_dataset(path):
    # return Dataset.load_from_disk(path)
    with open(path, 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def load_and_preprocess_datasets(my_tokenizer,config):
    train_save_path = config["train_save_path"]
    val_save_path = config["val_save_path"]
    test_save_path = config["test_save_path"]

    if os.path.exists(train_save_path) and os.path.exists(val_save_path) and os.path.exists(test_save_path):
        train_dataset = load_preprocessed_dataset(train_save_path)
        val_dataset = load_preprocessed_dataset(val_save_path)
        test_dataset = load_preprocessed_dataset(test_save_path)
    else:
#         train_dataset = load_dataset('csv', data_files=config["train_file"], cache_dir='/home/pod/shared-nvme/data/cachefile')['train']
#         val_dataset = load_dataset('csv', data_files=config["validation_file"], cache_dir='/home/pod/shared-nvme/data/cachefile')['train']
#         test_dataset = load_dataset('csv', data_files=config["test_file"], cache_dir='/home/pod/shared-nvme/data/cachefile')['train']
        train_dataset = load_dataset('csv', data_files=config["train_file"])['train']
        val_dataset = load_dataset('csv', data_files=config["validation_file"])['train']
        test_dataset = load_dataset('csv', data_files=config["test_file"])['train']

        train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, my_tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda examples: preprocess_function(examples, my_tokenizer), batched=True)
        test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, my_tokenizer), batched=True)

        save_dataset(train_dataset, train_save_path)
        save_dataset(val_dataset, val_save_path)
        save_dataset(test_dataset, test_save_path)

    return train_dataset, val_dataset, test_dataset

# train_dataset, val_dataset, test_dataset = load_and_preprocess_datasets()
# [1] Encapsulate data preprocessing, training, and evaluation
# logic in a training function
# ============================================================

# Custom callback definition
# class SaveTokenizerCallback(TrainerCallback):
#     def __init__(self, tokenizer, output_dir):
#         self.tokenizer = tokenizer
#         self.output_dir = output_dir

#     def on_train_end(self, args, state, control, **kwargs):
#         self.tokenizer.save_pretrained(self.output_dir)
#         print(f"Tokenizer saved to {self.output_dir}")
            
def train_func(config):
    ###################################* load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer = T5Tokenizer.from_pretrained(config["model_path"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_path"])

    ###################################* load data
    train_dataset, val_dataset, test_dataset = load_and_preprocess_datasets(tokenizer,config)

    ###################################* metrix
#     def compute_metrics(p):
#         predictions, labels = p

#         if isinstance(predictions, tuple):
#             predictions = predictions[0]
#         predictions = np.argmax(predictions, axis=-1)

#         predictions = predictions.flatten()
#         labels = labels.flatten()

#         metric = load_metric("accuracy")
#         return metric.compute(predictions=predictions, references=labels)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=32 // config["batch_size"],  # Simulate larger batch sizes
        num_train_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        save_total_limit=1,          # 只保留一个checkpoint
        load_best_model_at_end=True, # 是否在训练结束时加载最佳模型
        metric_for_best_model="eval_loss", # 选择最佳模型的指标
        greater_is_better=False,  # 表示更小的评估损失表示更好的模型
        save_strategy="epoch",  # 表示每个训练周期结束后保存模型
        eval_strategy="epoch", # 每个训练周期结束后评估模型
        # fp16=False,  # enable mixed precision training
        fp16=True,  # enable mixed precision training
    )
        
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        data_collator=data_collator,
        # callbacks=[ray.train.huggingface.transformers.RayTrainReportCallback()]
        # callbacks=[ray.train.huggingface.transformers.RayTrainReportCallback(), SaveTokenizerCallback(tokenizer, config["output_dir"])]

    )

    # trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()

    # 评估模型
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss")
    
    if val_loss is None:
        raise ValueError("Evaluation results did not contain 'eval_loss'")
    
    # 报告评估指标
    metrics = {"eval_loss": val_loss, "epoch": config["num_epochs"]}
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        trainer.save_model(temp_checkpoint_dir)
        trainer.save_state()
        trainer.save_metrics("eval", eval_results)
        tokenizer.save_pretrained(temp_checkpoint_dir)
        
        session.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        )
    
    if ray.train.get_context().get_world_rank() == 0:
        print(metrics)
    
    # # Save the trained model checkpoint
    # with tune.checkpoint_dir(step=trainer.state.global_step) as checkpoint_dir:
        # model.save_pretrained(checkpoint_dir)
        # tokenizer.save_pretrained(checkpoint_dir)

def tune_transformer(args):
    tune_config = {
        "model_path":args.model_path,
        "output_dir":args.output_dir,
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16]),
#         "batch_size": tune.choice([1,2,4]),
        "num_epochs": tune.choice([2, 3]),
        "weight_decay": tune.uniform(0.0, 0.3),
        "train_file":args.train_file,
        "validation_file":args.validation_file,
        "test_file":args.test_file,
        "train_save_path":args.traindata_process_save_path,
        "val_save_path":args.valdata_process_save_path,
        "test_save_path":args.testdata_process_save_path
    }

    scheduler = tune.schedulers.ASHAScheduler(
        # metric="eval_loss",
        # mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    
    # pbt = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     perturbation_interval=2,
    #     hyperparam_mutations={
    #         "learning_rate": tune.loguniform(1e-5, 1e-3),
    #         "per_device_train_batch_size": [4, 6, 8],
    #     }
    # )
        
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "num_train_epochs", "weight_decay"],
        metric_columns=["eval_accuracy", "eval_loss", "epoch", "training_iteration"],
        max_report_frequency=10,  # 控制报告频率
        print_intermediate_tables=False  # 关闭中间表格输出
    )
    analysis = tune.run(
        tune.with_parameters(train_func),
        metric="eval_loss",
        mode="min",
        resources_per_trial={"cpu": 19, "gpu":2},  # Adjust as needed
        # resources_per_trial={"cpu": 3, "gpu":0},  # Adjust as needed
        config=tune_config,
        num_samples=3,  # Number of trials
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_qa_model",
        storage_path=args.local_dir,   # 指定了存储调优结果的本地目录路径
        # local_dir=args.local_dir,   # 指定了存储调优结果的本地目录路径
        # stop={"training_iteration": 2},
        keep_checkpoints_num=3,  # 限制保存的 checkpoint 数量
    )

    # print("Best hyperparameters found were: ", analysis.best_config)
    print("Best hyperparameters found were: ", analysis.best_config)
    best_trial = analysis.get_best_trial(metric="eval_loss", mode="min")
    best_model_path = best_trial.checkpoint.path
    # 创建目标文件夹路径
    destination_path = args.bestmodel_dir
    # 复制整个目录
    print(best_model_path)
    shutil.copytree(best_model_path, destination_path)
    # model.save_pretrained(args.bestmodel_dir)
    print("The best model has been successfully saved as 'best_model'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # # 添加命令行参数
    parser.add_argument("--model_path", type=str, default = "flan-t5-small/", help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--train_file", type=str, default="data/mydata/mytrain.csv", help="The input training data file (a csv or json file).")
    parser.add_argument("--validation_file", type=str, default="data/mydata/myvalidation.csv", help="The input validation data file (a csv or json file).")
    parser.add_argument("--test_file", type=str, default="data/mydata/mytest.csv", help="The input test data file (a csv or json file).")
    parser.add_argument("--output_dir", type=str, default="results", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bestmodel_dir", type=str, default="best_model", help="The output directory where the best model will be saved in.")
    parser.add_argument("--local_dir", type=str, default="ray_results/", help="Specifies the local directory path where the tuning results are stored.")
    
    parser.add_argument("--traindata_process_save_path", type=str, default="data/mydata/preprocessed/train.pkl", help="Specifies the local directory path where the preprocessed data are stored.")
    parser.add_argument("--valdata_process_save_path", type=str, default="data/mydata/preprocessed/val.pkl", help="Specifies the local directory path where the preprocessed data are stored.")
    parser.add_argument("--testdata_process_save_path", type=str, default="data/mydata/preprocessed/test.pkl", help="Specifies the local directory path where the preprocessed data are stored.")
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True)
    tune_transformer(args)