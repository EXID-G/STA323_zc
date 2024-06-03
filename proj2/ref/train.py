import os

import torch
import numpy as np
import evaluate
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
    Seq2SeqTrainer,
)

import ray.train.huggingface.transformers
from ray.train import ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchCheckpoint

# [1] Encapsulate data preprocessing, training, and evaluation
# logic in a training function
# ============================================================
def train_func():
    # 加载数据
    dataset = load_dataset('json',data_files={'train':'/data/lab/project_2/data/train.json', 'validation':'/data/lab/project_2/data/validation.json'})
    
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
 
    
    # 加载预训练的模型和分词器
    model_name = "/data/lab/project_2/Task1/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 定义分词函数进行分词
    def tokenize_function(examples):
        model_inputs = tokenizer(examples["input_texts"], max_length=512, truncation=True)
        labels = tokenizer(examples["target_texts"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
    
    # 模型评估
    metric = evaluate.load("/data/lab/project_2/evaluate/squad_v2")
    
    def postprocess_qa_predictions(examples, features, predictions, stage="eval"):
        # This is a simple post-processing function to convert model predictions to the SQuAD format
        # You can replace it with more complex logic if needed
        return predictions
    
    def compute_metrics(p):
        predictions, labels = p
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)

        predictions = predictions.flatten()
        labels = labels.flatten()

        metric = load_metric("accuracy")
        return metric.compute(predictions=predictions, references=labels) 
    
    # 设置参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="/data/lab/project_2/results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=3,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 增加 CheckpointConfig(num_to_keep) 的值
    checkpoint_config = CheckpointConfig(
    num_to_keep=10,  
    checkpoint_frequency=5  
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer = tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #post_process_function=postprocess_qa_predictions,
    )

    # [2] Report Metrics and Checkpoints to Ray Train
    # ===============================================
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

        # [3] Prepare Transformers Trainer
    # ================================
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Start Training
    trainer.train()
    
# [4] Define a Ray TorchTrainer to launch `train_func` on all workers
# ===================================================================
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    # [4a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
result = ray_trainer.fit()

# [5] Load the trained model.
with result.checkpoint.as_directory() as checkpoint_dir:
    checkpoint_path = os.path.join(
        checkpoint_dir,
    ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
    )
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)    


