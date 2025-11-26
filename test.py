from llama_cpp import Llama
import os, sys, time

print("PID:", os.getpid())
print("ENV OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

model = "infrastructure/docker/models/Qwen3-8B-Q4_K_M.gguf"   # <-- sửa path ở đây

try:
    llm = Llama(
        model_path=model,
        n_ctx=1024,
        n_threads=1,
        n_batch=1,
        n_gpu_layers=0,
        use_mmap=False,
        use_mlock=False,
        verbose=True,
    )
    print("Loaded OK")
    start = time.time()
    res = llm(prompt="Hello", max_tokens=5, temperature=0.1)
    print("Response:", res)
    print("Elapsed:", time.time()-start)
except Exception as e:
    print("Exception:", e)
    raise
