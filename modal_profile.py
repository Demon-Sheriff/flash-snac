"""
Modal deployment for SNAC profiling on H100 GPUs.

Usage:
    # Run profiling remotely on H100
    modal run modal_profile.py

    # Deploy as a service
    modal deploy modal_profile.py
"""

import modal

app = modal.App("snac-profiler")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "triton==2.2.0",
        "einops",
        "huggingface_hub",
        "numpy",
    )
    .run_commands(
        "pip install --no-deps snac",  # Install snac without pulling deps again
    )
)

# Volume for caching models and results
volume = modal.Volume.from_name("snac-optimized", create_if_missing=True)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=1800,
)
def profile_snac(
    batch_size: int = 1,
    audio_seconds: float = 5.0,
    model_id: str = "hubertsiuzdak/snac_24khz",
    iterations: int = 200,
):
    """
    Profile SNAC decoder on H100 and return detailed results.
    """
    import torch
    import numpy as np
    from snac import SNAC
    import os
    import math

    # Set cache dir
    os.environ["HF_HOME"] = "/cache/huggingface"

    print(f"=== SNAC Profiler on H100 ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Load model
    print(f"Loading model: {model_id}")
    model = SNAC.from_pretrained(model_id).cuda().eval()
    print(f"Model config: sampling_rate={model.sampling_rate}, decoder_dim={model.decoder_dim}")
    print(f"VQ strides: {model.vq_strides}")
    print()

    # Generate test codes
    def generate_codes(batch_size, audio_seconds):
        samples = int(audio_seconds * model.sampling_rate)
        hop = model.hop_length
        attn_window = model.attn_window_size or 1
        lcm = math.lcm(model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to
        latent_len = padded_samples // hop

        codes = []
        for stride in model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(0, model.codebook_size, (batch_size, code_len), device="cuda")
            codes.append(code)
        return codes

    codes = generate_codes(batch_size, audio_seconds)
    print(f"Test codes generated for batch={batch_size}, audio={audio_seconds}s:")
    for i, c in enumerate(codes):
        print(f"  codes[{i}]: {tuple(c.shape)}")
    print()

    # Count Snake1d modules
    from snac.layers import Snake1d
    snake_count = 0
    for module in model.decoder.modules():
        if isinstance(module, Snake1d):
            snake_count += 1
    print(f"Snake1d modules in decoder: {snake_count}")

    # Count Snake1d forward calls
    call_count = [0]
    hooks = []
    for module in model.decoder.modules():
        if isinstance(module, Snake1d):
            def hook(m, i, o):
                call_count[0] += 1
            hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = model.decode(codes)

    for h in hooks:
        h.remove()

    print(f"Snake1d calls per decode: {call_count[0]}")
    print()

    # Warmup
    print(f"Warming up ({50} iterations)...")
    for _ in range(50):
        with torch.no_grad():
            _ = model.decode(codes)
    torch.cuda.synchronize()

    # Detailed timing
    print(f"Running detailed profiling...")
    vq_times = []
    decoder_times = []
    total_times = []

    for _ in range(iterations):
        with torch.no_grad():
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            vq_start = torch.cuda.Event(enable_timing=True)
            vq_end = torch.cuda.Event(enable_timing=True)
            dec_start = torch.cuda.Event(enable_timing=True)
            dec_end = torch.cuda.Event(enable_timing=True)

            total_start.record()

            vq_start.record()
            z_q = model.quantizer.from_codes(codes)
            vq_end.record()

            dec_start.record()
            audio = model.decoder(z_q)
            dec_end.record()

            total_end.record()
            torch.cuda.synchronize()

            vq_times.append(vq_start.elapsed_time(vq_end))
            decoder_times.append(dec_start.elapsed_time(dec_end))
            total_times.append(total_start.elapsed_time(total_end))

    vq_times = np.array(vq_times)
    decoder_times = np.array(decoder_times)
    total_times = np.array(total_times)

    results = {
        "gpu": torch.cuda.get_device_name(),
        "batch_size": batch_size,
        "audio_seconds": audio_seconds,
        "snake_modules": snake_count,
        "snake_calls_per_decode": call_count[0],
        "iterations": iterations,
        "vq_from_codes": {
            "mean_ms": float(np.mean(vq_times)),
            "std_ms": float(np.std(vq_times)),
            "pct_of_total": float(np.mean(vq_times) / np.mean(total_times) * 100),
        },
        "decoder_forward": {
            "mean_ms": float(np.mean(decoder_times)),
            "std_ms": float(np.std(decoder_times)),
            "pct_of_total": float(np.mean(decoder_times) / np.mean(total_times) * 100),
        },
        "total_decode": {
            "mean_ms": float(np.mean(total_times)),
            "std_ms": float(np.std(total_times)),
            "p50_ms": float(np.percentile(total_times, 50)),
            "p95_ms": float(np.percentile(total_times, 95)),
            "p99_ms": float(np.percentile(total_times, 99)),
        },
        "realtime_factor": float(audio_seconds * 1000 / np.mean(total_times)),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }

    # Print summary
    print()
    print("=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"GPU: {results['gpu']}")
    print(f"Batch: {batch_size}, Audio: {audio_seconds}s")
    print("-" * 60)
    print(f"{'Operation':<25} {'Mean (ms)':<12} {'% Total':<10}")
    print("-" * 60)
    print(f"{'vq_from_codes':<25} {results['vq_from_codes']['mean_ms']:>10.3f}  {results['vq_from_codes']['pct_of_total']:>8.1f}%")
    print(f"{'decoder_forward':<25} {results['decoder_forward']['mean_ms']:>10.3f}  {results['decoder_forward']['pct_of_total']:>8.1f}%")
    print("-" * 60)
    print(f"{'Total decode':<25} {results['total_decode']['mean_ms']:>10.3f}")
    print(f"{'P95 latency':<25} {results['total_decode']['p95_ms']:>10.3f}")
    print(f"{'P99 latency':<25} {results['total_decode']['p99_ms']:>10.3f}")
    print("-" * 60)
    print(f"Real-time factor: {results['realtime_factor']:.1f}x")
    print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    print("=" * 60)

    return results


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=1800,
)
def profile_pytorch_profiler(
    batch_size: int = 1,
    audio_seconds: float = 5.0,
    model_id: str = "hubertsiuzdak/snac_24khz",
):
    """
    Run PyTorch profiler for kernel-level analysis.
    Returns the profiler table as a string.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from snac import SNAC
    import os
    import math

    os.environ["HF_HOME"] = "/cache/huggingface"

    print(f"Loading model: {model_id}")
    model = SNAC.from_pretrained(model_id).cuda().eval()

    # Generate codes
    def generate_codes(batch_size, audio_seconds):
        samples = int(audio_seconds * model.sampling_rate)
        hop = model.hop_length
        attn_window = model.attn_window_size or 1
        lcm = math.lcm(model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to
        latent_len = padded_samples // hop

        codes = []
        for stride in model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(0, model.codebook_size, (batch_size, code_len), device="cuda")
            codes.append(code)
        return codes

    codes = generate_codes(batch_size, audio_seconds)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.decode(codes)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            with torch.no_grad():
                with record_function("snac_decode"):
                    _ = model.decode(codes)
            torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    print(table)

    return table


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=1800,
)
def benchmark_sweep(
    model_id: str = "hubertsiuzdak/snac_24khz",
):
    """
    Run a sweep of batch sizes and audio durations for comprehensive benchmarking.
    """
    import torch
    from snac import SNAC
    import os
    import math
    import numpy as np

    os.environ["HF_HOME"] = "/cache/huggingface"

    model = SNAC.from_pretrained(model_id).cuda().eval()

    def generate_codes(batch_size, audio_seconds):
        samples = int(audio_seconds * model.sampling_rate)
        hop = model.hop_length
        attn_window = model.attn_window_size or 1
        lcm = math.lcm(model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to
        latent_len = padded_samples // hop

        codes = []
        for stride in model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(0, model.codebook_size, (batch_size, code_len), device="cuda")
            codes.append(code)
        return codes

    def benchmark(codes, warmup=20, iterations=100):
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.decode(codes)
        torch.cuda.synchronize()

        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = model.decode(codes)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    # Test matrix - focused on real-time TTS (batch=1)
    test_cases = [
        (1, 1.0),   # Short utterance
        (1, 2.0),
        (1, 5.0),   # Typical sentence
        (1, 10.0),  # Long sentence
        (1, 30.0),  # Paragraph
    ]

    results = []
    print(f"{'Batch':<6} {'Audio(s)':<10} {'Mean(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'RTF':<8}")
    print("-" * 60)

    for batch_size, audio_seconds in test_cases:
        codes = generate_codes(batch_size, audio_seconds)
        stats = benchmark(codes)
        rtf = audio_seconds * 1000 / stats["mean_ms"]

        print(f"{batch_size:<6} {audio_seconds:<10.1f} {stats['mean_ms']:<12.3f} {stats['p50_ms']:<12.3f} {stats['p95_ms']:<12.3f} {rtf:<8.1f}x")

        results.append({
            "batch_size": batch_size,
            "audio_seconds": audio_seconds,
            **stats,
            "realtime_factor": rtf,
        })

    return results


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=1800,
)
def profile_memory_allocations(
    model_id: str = "hubertsiuzdak/snac_24khz",
    audio_durations: list = [1.0, 5.0, 10.0, 30.0],
):
    """
    Profile memory allocation patterns across different audio lengths.

    Uses PyTorch CUDA memory snapshot to capture:
    - Number of allocations per decode
    - Allocation sizes and patterns
    - Memory scaling with audio length

    Saves snapshots to /cache for download and visualization at pytorch.org/memory_viz
    """
    import torch
    from snac import SNAC
    import os
    import math
    import pickle

    os.environ["HF_HOME"] = "/cache/huggingface"

    print(f"=== Memory Allocation Profiler ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    audio_durations.extend([30]*100) # hardcoded test for 100 times of audio testing for decode
    model = SNAC.from_pretrained(model_id).cuda().eval()

    def generate_codes(batch_size, audio_seconds):

        samples = int(audio_seconds * model.sampling_rate)
        hop = model.hop_length
        attn_window = model.attn_window_size or 1
        lcm = math.lcm(model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to
        latent_len = padded_samples // hop

        codes = []
        for stride in model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(0, model.codebook_size, (batch_size, code_len), device="cuda")
            codes.append(code)
        return codes

    results = []
    snapshot_paths = []

    # Warmup (without recording)
    for _ in range(10):
        with torch.no_grad():
            _ = model.decode(codes)
    torch.cuda.synchronize()

    for duration in audio_durations:
        print(f"\n--- Profiling {duration}s audio ---")
        codes = generate_codes(1, duration)

        # Clear memory stats
        torch.cuda.reset_peak_memory_stats()
        # torch.cuda.empty_cache() # don't empty cache, test under load for peak allocations / predictable behavior

        # Start recording memory history
        # max_entries limits buffer size to avoid excessive memory use
        torch.cuda.memory._record_memory_history() # dont use max_entries=100000 for now

        # Run a single decode to capture allocation pattern
        with torch.no_grad():
            _ = model.decode(codes)
        torch.cuda.synchronize()

        # Capture snapshot
        snapshot_path = f"/cache/memory_snapshot_{duration}s.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_path)
        snapshot_paths.append(snapshot_path)

        # Stop recording
        torch.cuda.memory._record_memory_history(enabled=None)

        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Load and analyze snapshot
        with open(snapshot_path, 'rb') as f:
            snapshot = pickle.load(f)

        # Count allocation events
        alloc_count = 0
        free_count = 0
        total_allocated = 0

        if 'device_traces' in snapshot:
            for trace in snapshot.get('device_traces', []):
                for event in trace:
                    if event.get('action') == 'alloc':
                        alloc_count += 1
                        total_allocated += event.get('size', 0)
                    elif event.get('action') == 'free':
                        free_count += 1

        result = {
            "audio_seconds": duration,
            "peak_memory_mb": peak_memory,
            "allocation_count": alloc_count,
            "free_count": free_count,
            "total_allocated_mb": total_allocated / 1024 / 1024,
            "snapshot_path": snapshot_path,
        }
        results.append(result)

        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Allocations: {alloc_count}, Frees: {free_count}")
        print(f"  Snapshot saved: {snapshot_path}")

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY ALLOCATION SUMMARY")
    print("=" * 70)
    print(f"{'Duration':<12} {'Peak(MB)':<12} {'Allocs':<12} {'Frees':<12} {'Alloc/Free':<12}")
    print("-" * 70)

    for r in results:
        ratio = r['allocation_count'] / max(r['free_count'], 1)
        print(f"{r['audio_seconds']:<12.1f} {r['peak_memory_mb']:<12.1f} {r['allocation_count']:<12} {r['free_count']:<12} {ratio:<12.2f}")

    # Check scaling
    if len(results) >= 2:
        mem_ratio = results[-1]['peak_memory_mb'] / results[0]['peak_memory_mb']
        time_ratio = results[-1]['audio_seconds'] / results[0]['audio_seconds']
        print(f"\nMemory scaling: {results[0]['audio_seconds']}s → {results[-1]['audio_seconds']}s")
        print(f"  Time ratio: {time_ratio:.1f}x")
        print(f"  Memory ratio: {mem_ratio:.1f}x")
        print(f"  Scaling: {'~linear' if 0.8 < mem_ratio/time_ratio < 1.2 else 'non-linear'}")

    print("\n" + "=" * 70)
    print("SNAPSHOT FILES (download and view at pytorch.org/memory_viz):")
    for path in snapshot_paths:
        print(f"  {path}")
    print("=" * 70)

    return results


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=1800,
)
def profile_allocation_hotspots(
    model_id: str = "hubertsiuzdak/snac_24khz",
    audio_seconds: float = 5.0,
):
    """
    Identify allocation hotspots by analyzing stack traces.

    Shows which operations allocate the most memory during decode.
    """
    import torch
    from snac import SNAC
    import os
    import math
    import pickle
    from collections import defaultdict

    os.environ["HF_HOME"] = "/cache/huggingface"

    print(f"=== Allocation Hotspot Analysis ===")
    model = SNAC.from_pretrained(model_id).cuda().eval()

    def generate_codes(batch_size, audio_seconds):
        samples = int(audio_seconds * model.sampling_rate)
        hop = model.hop_length
        attn_window = model.attn_window_size or 1
        lcm = math.lcm(model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to
        latent_len = padded_samples // hop

        codes = []
        for stride in model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(0, model.codebook_size, (batch_size, code_len), device="cuda")
            codes.append(code)
        return codes

    codes = generate_codes(1, audio_seconds)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.decode(codes)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Record with stack traces
    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        stacks="python",  # Capture Python stack traces
    )

    with torch.no_grad():
        _ = model.decode(codes)
    torch.cuda.synchronize()

    snapshot_path = "/cache/memory_hotspots.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    # Analyze stack traces
    with open(snapshot_path, 'rb') as f:
        snapshot = pickle.load(f)

    # Aggregate by stack trace
    alloc_by_stack = defaultdict(lambda: {"count": 0, "total_bytes": 0})

    if 'device_traces' in snapshot:
        for trace in snapshot.get('device_traces', []):
            for event in trace:
                if event.get('action') == 'alloc':
                    # Get stack trace (simplified)
                    frames = event.get('frames', [])
                    if frames:
                        # Use top frame as key
                        top_frame = frames[0] if frames else {}
                        key = f"{top_frame.get('filename', 'unknown')}:{top_frame.get('line', 0)}"
                    else:
                        key = "unknown"

                    alloc_by_stack[key]["count"] += 1
                    alloc_by_stack[key]["total_bytes"] += event.get('size', 0)

    # Sort by total bytes
    sorted_allocs = sorted(
        alloc_by_stack.items(),
        key=lambda x: x[1]["total_bytes"],
        reverse=True
    )

    print(f"\nTop allocation sites for {audio_seconds}s audio:")
    print("-" * 80)
    print(f"{'Location':<50} {'Count':<10} {'Total (MB)':<15}")
    print("-" * 80)

    for loc, stats in sorted_allocs[:20]:
        mb = stats["total_bytes"] / 1024 / 1024
        print(f"{loc:<50} {stats['count']:<10} {mb:<15.2f}")

    print(f"\nFull snapshot saved to: {snapshot_path}")
    print("Visualize at: pytorch.org/memory_viz")

    return {
        "snapshot_path": snapshot_path,
        "top_allocations": sorted_allocs[:20],
    }


@app.function(
    volumes={"/cache": volume},
)
def download_snapshots():
    """
    List and return paths to memory snapshots for download.
    """
    import os

    snapshot_dir = "/cache"
    snapshots = []

    for f in os.listdir(snapshot_dir):
        if f.endswith(".pickle") and "memory" in f:
            path = os.path.join(snapshot_dir, f)
            size_mb = os.path.getsize(path) / 1024 / 1024
            snapshots.append({"path": path, "size_mb": size_mb})

    return snapshots


@app.local_entrypoint()
def main(
    mode: str = "profile",
    batch_size: int = 1,
    audio_seconds: float = 5.0,
):
    """
    Local entrypoint for running profiling.

    Args:
        mode: Profiling mode
            - "profile": Detailed timing breakdown
            - "pytorch": Kernel-level analysis with PyTorch profiler
            - "sweep": Benchmark across audio durations
            - "memory": Memory allocation patterns across durations
            - "hotspots": Identify allocation hotspots with stack traces
        batch_size: Batch size for inference
        audio_seconds: Audio duration in seconds
    """
    if mode == "profile":
        results = profile_snac.remote(
            batch_size=batch_size,
            audio_seconds=audio_seconds,
        )
        print("\nReturned results:", results)

    elif mode == "pytorch":
        table = profile_pytorch_profiler.remote(
            batch_size=batch_size,
            audio_seconds=audio_seconds,
        )
        print("\nKernel-level profiling complete.")

    elif mode == "sweep":
        results = benchmark_sweep.remote()
        print("\nBenchmark sweep complete.")
        for r in results:
            print(r)

    elif mode == "memory":
        results = profile_memory_allocations.remote(
            audio_durations=[1.0, 5.0, 10.0, 30.0],
        )
        print("\nMemory profiling complete.")
        print("Snapshots saved to /cache - download and visualize at pytorch.org/memory_viz")

    elif mode == "hotspots":
        results = profile_allocation_hotspots.remote(
            audio_seconds=audio_seconds,
        )
        print("\nHotspot analysis complete.")
        print(f"Snapshot: {results['snapshot_path']}")

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes:")
        print("  profile   - Detailed timing breakdown (vq_from_codes vs decoder)")
        print("  pytorch   - Kernel-level analysis with PyTorch profiler")
        print("  sweep     - Benchmark across audio durations (1s, 5s, 10s, 30s)")
        print("  memory    - Memory allocation patterns across durations")
        print("  hotspots  - Identify allocation hotspots with stack traces")
