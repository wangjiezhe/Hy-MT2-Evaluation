import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from llama_cpp import GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, Llama
from openai import OpenAI
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

USER_PROMPT = """Translate the following text into Chinese.
Note that you should **only output the translated result without any additional explanation**:

"""


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
            flash_attn=True,
            offload_kqv=True,
            type_k=self.type_kv,
            type_v=self.type_kv,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm.close()

    def translate(self, text, prompt=USER_PROMPT):
        output = self.llm(
            prompt + text,
            temperature=0,
        )
        return output["choices"][0]["text"].strip()

    def translate_v1(self, text, prompt=USER_PROMPT, system_prompt=True):
        message = (
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
            if system_prompt
            else [
                {"role": "user", "content": prompt + text},
            ]
        )
        response = self.llm.create_chat_completion_openai_v1(
            messages=message,
            temperature=0,
        )
        return response.choices[0].message.content


def load_wmt24pp():
    ds = load_dataset("google/wmt24pp", name="en-zh_CN", split="train")
    sources = [item["source"] for item in ds if not item["is_bad_source"]]
    targets = [[item["target"]] for item in ds if not item["is_bad_source"]]
    return sources, targets


def evaluate_llama():
    sources, targets = load_wmt24pp()
    eng_scores = ""

    for quant in ["2B", "9B"]:
        for cache_type in ["F16", "Q8_0", "Q4_0"]:
            with LlamaModel(MODEL_PATH[quant], TYPE_KV[cache_type]) as model:
                # predictions = [model.translate(source) for source in tqdm(sources)]
                predictions = [model.translate_v1(source) for source in tqdm(sources)]
                bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
                chrf_score = corpus_chrf(predictions, targets, word_order=2)
                score = f"{MODEL_NAME[quant]}\t{cache_type}\t{bleu_score}\t{chrf_score}"
                print(score)
                eng_scores += f"{score}\n"

    print("\nTranslate from English to Chinese:")
    print(eng_scores)


def evaluate2_llama():
    sources, targets = load_wmt24pp()
    eng_scores = ""

    quant = "9B"
    zh_prompt = "将以下文本翻译为中文，注意**只需要输出翻译后的结果，不要额外解释**：\n"

    for use_system in [False, True]:
        for cache_type in ["F16", "Q8_0", "Q4_0"]:
            with LlamaModel(MODEL_PATH[quant], TYPE_KV[cache_type]) as model:
                predictions = [
                    model.translate_v1(
                        source, prompt=zh_prompt, system_prompt=use_system
                    )
                    for source in tqdm(sources)
                ]
                bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
                chrf_score = corpus_chrf(predictions, targets, word_order=2)
                score = f"{MODEL_NAME[quant]}\t{cache_type}\t{bleu_score}\t{chrf_score}"
                print(score)
                eng_scores += f"{score}\n"

    print("\nTranslate from English to Chinese:")
    print(eng_scores)


def evaluate3_llama():
    sources, targets = load_wmt24pp()
    eng_scores = ""

    quant = "9B"
    cache_type = "Q4_0"
    for use_system in [False, True]:
        with LlamaModel(MODEL_PATH[quant], TYPE_KV[cache_type]) as model:
            predictions = [
                model.translate_v1(source, system_prompt=use_system)
                for source in tqdm(sources)
            ]
            bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
            chrf_score = corpus_chrf(predictions, targets, word_order=2)
            score = f"{MODEL_NAME[quant]}\t{cache_type}\t{bleu_score}\t{chrf_score}"
            print(score)
            eng_scores += f"{score}\n"

    print("\nTranslate from English to Chinese:")
    print(eng_scores)


def evaluate_vllm():
    sources, targets = load_wmt24pp()
    predictions = []

    client = OpenAI(base_url="http://localhost:8118/v1", api_key="EMPTY")

    for source in tqdm(sources):
        response = client.chat.completions.create(
            model="IndexTeam/Index-Translate-2B",
            messages=[
                # {"role": "user", "content": USER_PROMPT + source},
                {"role": "system", "content": USER_PROMPT},
                {"role": "user", "content": source},
            ],
        )
        predictions.append(response.choices[0].message.content)

    bleu_score = corpus_bleu(predictions, targets, tokenize="zh")
    chrf_score = corpus_chrf(predictions, targets, word_order=2)
    print(f"IndexTeam/Index-Translate-2B\t{bleu_score}\t{chrf_score}")


def evaluate_lmstudio(model, max_workers: int = 4):
    sources, targets = load_wmt24pp()
    # sources = sources[:10]
    # targets = targets[:10]
    predictions = [None] * len(sources)

    def get_client():
        if not hasattr(_local, "client"):
            _local.client = OpenAI(
                base_url="http://localhost:1234/v1",
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
                {"role": "system", "content": USER_PROMPT},
                {"role": "user", "content": source},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
        print(f"{model}\t{bleu_score}\t{chrf_score}")


if __name__ == "__main__":
    # evaluate_llama()
    # evaluate_vllm()
    # evaluate2_llama()
    # evaluate3_llama()
    # evaluate_lmstudio("IndexTeam/Index-Translate-2B", 8)
    evaluate_lmstudio("IndexTeam/Index-Translate-9B", 4)
