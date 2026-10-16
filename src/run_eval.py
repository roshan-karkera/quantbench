"""
Run one precision configuration of a model over the fixed task set and record
quality alongside latency, throughput and peak memory. One invocation is one
MLflow run.

    python -m src.run_eval --precision fp16
    python -m src.run_eval --precision int8
    python -m src.run_eval --precision nf4
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "eval" / "tasks.jsonl"
RESULTS = ROOT / "results"


def load_tasks():
    with open(TASKS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_quant_config(precision):
    """Return (quantization_config, torch_dtype) for a named precision."""
    if precision == "fp16":
        return None, torch.float16
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), torch.float16
    if precision == "nf4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            torch.float16,
        )
    raise ValueError(f"unknown precision: {precision}")


def extract_code(text):
    """Pull the first python block out of a completion, else return it whole."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def normalise(text):
    """Reduce a short answer to something comparable."""
    text = text.strip().split("\n")[0].strip()
    text = text.strip("`*. ").replace(",", "")
    return text.lower()


def candidates(text):
    """Pull every plausible short answer out of a completion, last first.

    An instruct model is asked for a bare value and usually writes a sentence
    of reasoning first, so grading only the first line grades the wrong line.
    Extracting candidates and accepting any match is the standard fix, and the
    tasks are written so that only one value in the response can be the answer.
    """
    out = []
    out += re.findall(r"\b\d{1,2}:\d{2}\b", text)              # 12:15
    out += re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))  # 13.5, 180
    out += re.findall(r"\b(?:yes|no)\b", text, re.IGNORECASE)  # yes / no
    return list(reversed(out))


def score_exact(completion, expected):
    want = normalise(expected)

    # the clean case: the model answered with the bare value
    if normalise(completion) == want:
        return True

    for got in candidates(completion):
        got = got.strip().lower()
        if got == want:
            return True
        try:
            if abs(float(got) - float(want)) < 1e-6:
                return True
        except ValueError:
            continue
    return False


def score_python(completion, tests):
    """Execute the generated function against the task's assertions.

    This runs model generated code. Run it in a container or a throwaway VM,
    never on a machine you care about.
    """
    namespace = {}
    try:
        exec(extract_code(completion), namespace)  # noqa: S102
        exec(tests, namespace)  # noqa: S102
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", required=True, choices=["fp16", "int8", "nf4"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--experiment", default="quantbench")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for meaningful latency and memory numbers.")

    tasks = load_tasks()
    quant_config, dtype = build_quant_config(args.precision)

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="cuda:0",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    weights_mib = torch.cuda.max_memory_allocated() / 1024**2

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.model.split('/')[-1]}-{args.precision}"):
        mlflow.log_params(
            {
                "model": args.model,
                "precision": args.precision,
                "max_new_tokens": args.max_new_tokens,
                "greedy": True,
                "n_tasks": len(tasks),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        )

        records = []
        correct = {"reasoning": 0, "coding": 0}
        totals = {"reasoning": 0, "coding": 0}
        gen_tokens = 0
        gen_seconds = 0.0
        ttft_samples = []

        # one untimed warmup so the first real task is not paying for kernels
        warm = tokenizer("warmup", return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model.generate(**warm, max_new_tokens=8, do_sample=False)

        torch.cuda.reset_peak_memory_stats()

        for task in tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                first = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            ttft_samples.append(time.perf_counter() - start)
            del first

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
            completion = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )

            gen_tokens += int(new_tokens)
            gen_seconds += elapsed

            if task["check"] == "exact":
                ok = score_exact(completion, task["answer"])
            else:
                ok = score_python(completion, task["tests"])

            totals[task["type"]] += 1
            correct[task["type"]] += int(ok)

            records.append(
                {
                    "id": task["id"],
                    "type": task["type"],
                    "correct": ok,
                    "seconds": round(elapsed, 4),
                    "new_tokens": int(new_tokens),
                    "completion": completion,
                }
            )
            print(f"{task['id']:<10} {task['type']:<9} {'pass' if ok else 'FAIL'}  {elapsed:6.2f}s")

        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        n = len(tasks)
        overall = sum(correct.values()) / n

        metrics = {
            "accuracy_overall": overall,
            "accuracy_reasoning": correct["reasoning"] / max(totals["reasoning"], 1),
            "accuracy_coding": correct["coding"] / max(totals["coding"], 1),
            "tokens_per_second": gen_tokens / gen_seconds,
            "latency_mean_s": gen_seconds / n,
            "ttft_mean_s": sum(ttft_samples) / len(ttft_samples),
            "peak_memory_mib": peak_mib,
            "weights_memory_mib": weights_mib,
            "load_seconds": load_seconds,
        }
        mlflow.log_metrics(metrics)

        RESULTS.mkdir(exist_ok=True)
        out_path = RESULTS / f"{args.precision}.json"
        payload = {"config": vars(args), "metrics": metrics, "records": records}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_path))

        print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
