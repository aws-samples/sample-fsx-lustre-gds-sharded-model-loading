import time
import sys

def main():
    from vllm import LLM, SamplingParams

    model = sys.argv[1] if len(sys.argv) > 1 else "/fsx/models/Llama-3.1-405B"
    load_format = sys.argv[2] if len(sys.argv) > 2 else "auto"
    quant = sys.argv[3] if len(sys.argv) > 3 else "fp8"

    print(f"Model: {model}")
    print(f"Load format: {load_format}")
    print(f"Quantization: {quant}")

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        load_format=load_format,
        quantization=quant if quant != "none" else None,
        tensor_parallel_size=8,
        max_model_len=1024,
    )
    t1 = time.perf_counter()

    print(f"\n=== RESULT ===")
    print(f"Model loaded in {t1 - t0:.2f} seconds ({(t1-t0)/60:.1f} minutes)")

    output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"Sanity check: {output[0].outputs[0].text}")

if __name__ == "__main__":
    main()
