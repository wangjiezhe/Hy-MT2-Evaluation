import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from llama_cpp import GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, Llama
from openai import OpenAI, omit
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm

_local = threading.local()

MODEL_PATH = {
    "2B": (
        "/mnt/c/Users/wangjiezhe/.lmstudio/models/"
        "IndexTeam/Index-Translate-2B-GGUF/Index-Translate-2B.Q8_0.gguf"
    ),
    "9B": (
        "/mnt/c/Users/wangjiezhe/.lmstudio/models/"
        "IndexTeam/Index-Translate-9B-GGUF/Index-Translate-9B.Q4_K_M.gguf"
    ),
}

MODEL_NAME = {"2B": "Index-Translate-2B:Q8_0", "9B": "Index-Translate-9B:Q4_K_M"}

TYPE_KV = {"F16": GGML_TYPE_F16, "Q8_0": GGML_TYPE_Q8_0, "Q4_0": GGML_TYPE_Q4_0}

EN_PROMPT = """Translate the following text into Chinese.
Note that you should **only output the translated result without any additional explanation**:
"""
ZH_PROMPT = "将以下文本翻译为中文，注意**只需要输出翻译后的结果，不要额外解释**：\n"

NEW_EN_PROMPT = (
    "Translate the following text into Chinese. "
    "Output the translation directly, without any explanation:\n"
)
NEW_ZH_PROMPT = "请将以下文本翻译为中文，直接输出翻译结果，不要进行任何解释：\n"


class LlamaModel:
    def __init__(self, model_path, type_kv=GGML_TYPE_F16):
        self.model_path = model_path
        self.type_kv = type_kv

    def __enter__(self):
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=12,
            n_gpu_layers=-1,
            verbose=False,
            use_mmap=True,
            use_mlock=True,
            flash_attn=True,
            offload_kqv=True,
            type_k=self.type_kv,
            type_v=self.type_kv,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm.close()

    def translate(self, text, prompt=EN_PROMPT):
        output = self.llm(
            prompt + text,
            temperature=0,
        )
        pred = output["choices"][0]["text"].strip()
        if __debug__:
            print(text)
            print(pred)
            print()
        return pred

    def translate_v1(self, text, prompt=EN_PROMPT, system_prompt=True):
        message = (
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
            if system_prompt
            else [
                {"role": "user", "content": prompt + "\n" + text},
            ]
        )
        response = self.llm.create_chat_completion_openai_v1(
            messages=message,
            temperature=0,
        )
        pred = response.choices[0].message.content
        if __debug__:
            print(text)
            print(pred)
            print()
        return pred


def load_wmt24pp():
    ds = load_dataset("google/wmt24pp", name="en-zh_CN", split="train")
    sources = [item["source"] for item in ds if not item["is_bad_source"]]
    targets = [[item["target"]] for item in ds if not item["is_bad_source"]]
    return sources, targets


def evaluate_llama():
    sources, targets = load_wmt24pp()
    eng_scores = ""

    for quant in ["2B"]:  # , "9B"]:
        for cache_type in ["F16", "Q8_0", "Q4_0"]:
            with LlamaModel(MODEL_PATH[quant], TYPE_KV[cache_type]) as model:
                # predictions = [
                #     model.translate(source, prompt=ZH_PROMPT)
                #     for source in tqdm(sources)
                # ]
                predictions = [
                    model.translate_v1(source, prompt=EN_PROMPT, system_prompt=False)
                    for source in tqdm(sources)
                ]
                bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
                chrf_score = corpus_chrf(predictions, targets, word_order=2)
                score = f"{MODEL_NAME[quant]}\t{cache_type}\t{bleu_score}\t{chrf_score}"
                print(score)
                eng_scores += f"{score}\n"

    print("\nTranslate from English to Chinese:")
    print(eng_scores)


def evaluate2_llama():
    sources, targets = load_wmt24pp()
    sources = sources[180:185]
    targets = targets[180:185]
    eng_scores = ""

    quant = "9B"
    cache_type = "F16"
    with LlamaModel(MODEL_PATH[quant], TYPE_KV[cache_type]) as model:
        predictions = [
            model.translate_v1(source, prompt=EN_PROMPT, system_prompt=True)
            for source in sources
        ]
        bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
        chrf_score = corpus_chrf(predictions, targets, word_order=2)
        score = f"{MODEL_NAME[quant]}\t{cache_type}\t{bleu_score}\t{chrf_score}"
        print(score)
        eng_scores += f"{score}\n"

    print("\nTranslate from English to Chinese:")
    print(eng_scores)


def evaluate_server(
    model,
    base_url,
    max_workers,
    temperature=omit,
    frequency_penalty=omit,
    repetition_penalty=1.0,
    system_prompt=ZH_PROMPT,
):
    sources, targets = load_wmt24pp()
    if __debug__:
        sources = sources[545:555]
        targets = targets[545:555]
    predictions = [None] * len(sources)

    def get_client():
        if not hasattr(_local, "client"):
            _local.client = OpenAI(
                base_url=base_url,
                api_key="EMPTY",
                timeout=300.0,
            )
        return _local.client

    def translate(args):
        idx, source = args
        client = get_client()  # 线程内复用
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source},
            ],
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            seed=42,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "repetition_penalty": repetition_penalty,
            },
        )
        return idx, response.choices[0].message.content

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map 保证按输入顺序产出，进度条可用
            for idx, pred in tqdm(
                executor.map(translate, enumerate(sources)),
                total=len(sources),
            ):
                predictions[idx] = pred
                if __debug__:
                    print(f"{idx}:")
                    print(sources[idx])
                    print(pred)
                    print(targets[idx])
                    print()
    except KeyboardInterrupt:
        print("\nInterrupted! Calculating scores on available predictions...")
    finally:
        valid = [(p, t) for p, t in zip(predictions, targets) if p is not None]
        if not valid:
            print("No predictions to score.")
            sys.exit(1)
        preds, tgts = zip(*valid)

        bleu_score = corpus_bleu(preds, tgts, tokenize="zh")
        chrf_score = corpus_chrf(preds, tgts, word_order=2)
        print(f"{model}\t{bleu_score}\t{chrf_score}\t\tlength = {len(preds)}")


def evaluate_lmstudio(model, max_workers=4):
    return evaluate_server(model, "http://localhost:1234/v1", max_workers)


def evaluate_vllm(model, max_workers=128, temperature=0):
    return evaluate_server(
        model,
        "http://localhost:8118/v1",
        max_workers,
        temperature,
        frequency_penalty=0.4,
        repetition_penalty=1.1,
    )


if __name__ == "__main__":
    # evaluate_llama()
    # evaluate2_llama()
    # evaluate_vllm("IndexTeam/Index-Translate-2B", 256)
    # evaluate_vllm("IndexTeam/Index-Translate-2B", 128)
    evaluate_vllm("IndexTeam/Index-Translate-2B", 64)
    # evaluate_vllm("IndexTeam/Index-Translate-2B", 32)
    # evaluate_vllm("IndexTeam/Index-Translate-2B", 1)
    # evaluate_lmstudio("IndexTeam/Index-Translate-2B", 8)
    # evaluate_lmstudio("IndexTeam/Index-Translate-9B", 4)
