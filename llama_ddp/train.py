# train.py
import os
import argparse
import getpass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch.distributed as dist


def main(args):

    # Torchrun configs
    print("==== Torchrun Env Variables ====")
    for var in ["RANK", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        print(f"{var} = {os.environ.get(var)}\n")

    # Get Paths
    user_id = getpass.getuser()
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    save_dir = os.path.join(args.save_dir_base, user_id, "output", job_id)
    os.makedirs(save_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Load dataset and process
    def tokenize_function(data):
        return tokenizer(data["text"], truncation=True, padding="max_length", max_length=128)
    
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train[:1%]")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda x: {'labels': x['input_ids']}, batched=True) # it will shift one step to the right during training

    # Training args
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(save_dir)
    print(f"Train complete. Model saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_dir_base", type=str, default="/scratch3")

    args = parser.parse_args()
    main(args)