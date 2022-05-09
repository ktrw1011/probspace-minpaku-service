import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import partial

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from datasets import Dataset
from transformers import (
    set_seed,
    AutoModelForSequenceClassification, AutoTokenizer,
    AutoConfig, DataCollatorWithPadding,
    Trainer, TrainingArguments, EvalPrediction,
)

@dataclass
class Config:
    seed:int=42
    exp_name:str="exp015"
    max_length:int=512
    backbone_name:str="xlm-roberta-base"

cfg = Config()

def load_cv(path:str):
    with open(path, "rb") as f:
        return pickle.load(f)

def rmsle(y, pred):
    return np.sqrt(mean_squared_log_error(y, pred))

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds).astype(np.float32)
    return {"mse": ((preds - p.label_ids) ** 2).mean().item()}

def preprocess_function(examples, tokenizer, max_length:int):
    sep_token = tokenizer.sep_token
    input_text = "number of reviews, " + str(examples["number_of_reviews"]) + sep_token +\
        "minimum nights, " + str(examples["minimum_nights"]) + sep_token +\
            examples["room_type"] + sep_token + examples["neighbourhood"] + sep_token + examples["name"]

    input_text = input_text.lower()

    result = tokenizer(input_text, max_length=max_length, truncation=True)

    return result

def inverse_transform(encoder: MinMaxScaler, preds:np.ndarray):
    if len(preds.shape) != 2:
        preds = preds.reshape(-1, 1)
    preds = encoder.inverse_transform(preds).squeeze()
    return preds

def main():
    train_df = pd.read_csv("../input/train_data.csv")
    test_df = pd.read_csv("../input/test_data.csv")

    train_df["label"] = train_df["y"].copy()
    train_df["label"] = train_df["label"].map(np.log1p)

    min_max_encoder = MinMaxScaler().fit(train_df[["label"]])
    train_df["label"] = min_max_encoder.transform(train_df[["label"]]).reshape(-1)

    for col in ["name", "neighbourhood", "room_type"]:
        train_df[col] = train_df[col].str.lower()
        test_df[col] = test_df[col].str.lower()

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)

    preprocess_function_func = partial(preprocess_function, tokenizer=tokenizer, max_length=cfg.max_length)

    train_ds = train_ds.map(
        preprocess_function_func,
        desc="Running tokenizer on dataset"
    )

    test_ds = test_ds.map(
        preprocess_function_func,
        desc="Running tokenizer on dataset",
    )

    train_ds.set_format(columns=["input_ids", "attention_mask", "label"], type="numpy")
    test_ds.set_format(columns=["input_ids", "attention_mask"], type="numpy")
    sub_df = pd.read_csv("../input/submission.csv")

    cv = load_cv("../input/fold/kfold.pkl")

    oof_val_preds = np.zeros((len(train_ds)), dtype=np.float32)
    oof_test_preds = np.zeros((5, len(test_ds)), dtype=np.float32)

    for fold_idx, (trn_idx, val_idx) in enumerate(cv):
        print(f"=== fold {fold_idx} start ===")
        fold_name = f"fold_{fold_idx}"

        fold_dir = Path("../output") / cfg.exp_name / fold_name

        trn_ds = train_ds.select(trn_idx)
        val_ds = train_ds.select(val_idx)

        training_args = TrainingArguments(
                output_dir=fold_dir,
                overwrite_output_dir="True",
                do_train=True,
                do_eval=True,
                do_predict=False,
                evaluation_strategy="steps",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                num_train_epochs=10,
                warmup_ratio=0.1,
                save_total_limit=1,
                logging_steps=100,
                eval_steps=100,
                save_steps=100,
                eval_delay=1500,
                load_best_model_at_end=True,
                metric_for_best_model="eval_mse",
                greater_is_better=False,
        )

        model_config = AutoConfig.from_pretrained(cfg.backbone_name, num_labels=1)
        model_config.attention_probs_dropout_prob = 0
        model_config.hidden_dropout_prob = 0
        model = AutoModelForSequenceClassification.from_pretrained(cfg.backbone_name, config=model_config)

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=trn_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        train_result = trainer.train()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        val_preds = trainer.predict(val_ds).predictions.astype(np.float32)
        # min max invers
        val_preds = inverse_transform(min_max_encoder, val_preds)
        # np.log1p => np.exp1m
        val_preds = np.expm1(val_preds)
        oof_val_preds[val_idx] = val_preds
        print(f"fold {fold_idx} acc: {rmsle(val_ds['y'], val_preds)}")

        test_preds = trainer.predict(test_ds).predictions.astype(np.float32)
        test_preds = inverse_transform(min_max_encoder, test_preds)
        test_preds = np.expm1(test_preds)
        oof_test_preds[fold_idx] = test_preds

    oof_test_preds = oof_test_preds.mean(axis=0)

    with open(f"../output/{cfg.exp_name}/oof_val_preds.pkl", "wb") as f:
        pickle.dump(oof_val_preds, f)

    with open(f"../output/{cfg.exp_name}/oof_test_preds.pkl", "wb") as f:
        pickle.dump(oof_test_preds, f)

    sub_df['y'] = test_preds
    sub_df.to_csv(f"../output/{cfg.exp_name}/sub.csv", index=False)

    print(f"overall acc: {rmsle(train_df['y'].values, oof_val_preds)}")

if __name__ == "__main__":
    set_seed(cfg.seed)
    main()