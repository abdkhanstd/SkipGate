"""Download the models of stages 5 and 6 to a local folder.

The language models come from ModelScope (the Llama models from its LLM-Research mirrors,
the same source that was used on the server), the small sentence encoder from the Hugging
Face Hub.
Usage: python download_models.py --dir ../../build/models [--only qwen1.5b llama3b ...]
"""
import argparse
import os

MODELS = {
    "qwen1.5b": ("ms", "Qwen/Qwen2.5-1.5B-Instruct"),
    "llama3b": ("ms", "LLM-Research/Llama-3.2-3B-Instruct"),
    "qwen7b": ("ms", "Qwen/Qwen2.5-7B-Instruct"),
    "llama8b": ("ms", "LLM-Research/Meta-Llama-3.1-8B-Instruct"),
    "bge": ("ms", "BAAI/bge-base-en-v1.5"),
    "minilm": ("hf", "sentence-transformers/all-MiniLM-L6-v2"),
}
SKIP = ["original/*", "*.gguf", "*.pth", "onnx/*", "*.onnx", "openvino/*", "*.bin", "*.h5", "*.ot",
        "*.msgpack"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for tag, (src, mid) in MODELS.items():
        if args.only and tag not in args.only:
            continue
        out = os.path.join(args.dir, tag)
        if os.path.exists(os.path.join(out, ".done")):
            print("have", tag, flush=True)
            continue
        if src == "hf":
            from huggingface_hub import snapshot_download
            snapshot_download(mid, local_dir=out, ignore_patterns=SKIP)
        else:
            from modelscope import snapshot_download
            snapshot_download(mid, local_dir=out,
                              ignore_file_pattern=[r"original/.*", r".*\.gguf", r".*\.pth", r"onnx/.*",
                                                   r".*\.onnx", r".*\.bin", r"openvino/.*"])
        open(os.path.join(out, ".done"), "w").write(mid)
        print("ready", tag, out, flush=True)
    print("DL_DONE", flush=True)


if __name__ == "__main__":
    main()
