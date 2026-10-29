"""Pre-train a compact Hindi BERT (masked language modelling) from scratch.

Sized for an 8 GB GPU. Resumable: rerunning continues from the newest checkpoint in --out.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch
from datasets import load_dataset
from transformers import (BertConfig, BertForMaskedLM, BertTokenizerFast,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

SIZES = {  # layers, hidden, heads, intermediate
    "small": (4, 512, 8, 2048),
    "medium": (8, 512, 8, 2048),
    "base": (12, 768, 12, 3072),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="data/bert_cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    tok = BertTokenizerFast.from_pretrained(args.tokenizer)
    layers, hidden, heads, inter = SIZES[args.size]
    config = BertConfig(vocab_size=tok.vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                        num_attention_heads=heads, intermediate_size=inter,
                        max_position_embeddings=512, pad_token_id=tok.pad_token_id)

    ds = load_dataset("text", data_files=args.corpus, split="train", cache_dir=args.cache)
    if args.max_lines:
        ds = ds.select(range(min(args.max_lines, len(ds))))
    ds = ds.map(lambda b: tok(b["text"], add_special_tokens=False), batched=True,
                remove_columns=["text"], num_proc=args.workers)
    block = args.seq_len - 2

    def pack(batch):
        ids = [t for seq in batch["input_ids"] for t in seq]
        n = len(ids) // block * block
        chunks = [ids[i:i + block] for i in range(0, n, block)]
        return {"input_ids": [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]}

    ds = ds.map(pack, batched=True, remove_columns=ds.column_names, num_proc=args.workers)
    ds = ds.train_test_split(test_size=2000, seed=args.seed)
    print(f"train blocks: {len(ds['train']):,}  eval blocks: {len(ds['test']):,}  "
          f"params: {sum(p.numel() for p in BertForMaskedLM(config).parameters()):,}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out, max_steps=args.steps, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_steps=min(10_000, args.steps // 10), weight_decay=0.01,
        lr_scheduler_type="linear", bf16=True, logging_steps=200, eval_strategy="steps",
        eval_steps=5000, save_steps=5000, save_total_limit=3, dataloader_num_workers=4,
        seed=args.seed, report_to=[])
    trainer = Trainer(model=BertForMaskedLM(config), args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["test"],
                      data_collator=DataCollatorForLanguageModeling(tok, mlm_probability=0.15))
    resume = bool(glob.glob(os.path.join(args.out, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print(trainer.evaluate(), flush=True)


if __name__ == "__main__":
    main()
