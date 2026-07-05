"""QLoRA fine-tune of TranslateGemma on the khutbah/Quran ar->de dataset.

Designed for Kaggle T4 x2 (12B) or a single T4 (4B dry run). Resumable:
checkpoints push to the HF Hub every save so a dying session loses nothing.

Usage (Kaggle, after `huggingface-cli login` / HF_TOKEN env):
  python training/finetune_qlora.py \
      --model google/translategemma-12b-it \
      --hub-repo <user>/translategemma-12b-khutbah-lora
  # dry run: --model google/translategemma-4b-it --hub-repo <user>/tg4b-khutbah-lora

After training: merge + GGUF conversion (see notebooks/finetune_kaggle.ipynb).
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]

def build_texts(tokenizer, max_len):
    """TranslateGemma's chat template requires structured user content
    (type/source_lang_code/target_lang_code) and renders the official
    translation prompt itself — so training matches inference exactly."""
    rows = [json.loads(l) for l in
            (ROOT / "training" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]

    def user_msg(ar):
        return {"role": "user", "content": [{
            "type": "text",
            "source_lang_code": "ar",
            "target_lang_code": "de-DE",
            "text": ar,
        }]}

    # the template may want the assistant turn as a plain string or as the
    # same structured form depending on version — probe once
    probe = rows[0]
    try:
        tokenizer.apply_chat_template(
            [user_msg(probe["ar"]), {"role": "assistant", "content": probe["de"]}],
            tokenize=False)
        def assistant_msg(de):
            return {"role": "assistant", "content": de}
    except Exception:
        def assistant_msg(de):
            return {"role": "assistant", "content": [{"type": "text", "text": de}]}

    def to_text(row):
        messages = [user_msg(row["ar"]), assistant_msg(row["de"])]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

    def tok(batch):
        # labels are produced (padded, -100 masked) by the LM data collator
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    # 2% held-out val split: eval_loss during training flags overfitting
    # early. The real quality benchmark stays tests/eval_translation.py.
    return ds.train_test_split(test_size=0.02, seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/translategemma-12b-it")
    ap.add_argument("--hub-repo", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=200)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    splits = build_texts(tokenizer, args.max_len)

    out_dir = str(ROOT / "training" / "out")
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=False, fp16=True,          # T4 has no bf16
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=args.hub_repo,
        hub_strategy="checkpoint",      # survive session death
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    resume = any(Path(out_dir).glob("checkpoint-*")) if Path(out_dir).exists() else False
    trainer = Trainer(model=model, args=targs,
                      train_dataset=splits["train"],
                      eval_dataset=splits["test"],
                      processing_class=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.push_to_hub()
    print("done — adapter on the Hub:", args.hub_repo)


if __name__ == "__main__":
    main()
