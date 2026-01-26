"""
Memory Stability Profiler for SNAC Decoder

Tests memory behavior under sustained load to detect:
- Memory leaks (growth over time)
- Allocation pattern stability
- Fragmentation issues

Usage:
    modal run modal_memory_stability.py
"""

import modal

app = modal.App("snac-memory-stability")

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
        "pip install --no-deps snac",
    )
)

volume = modal.Volume.from_name("snac-optimized", create_if_missing=True)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=3600,
)
def profile_memory_stability(
    audio_seconds: float = 15.0,
    total_iterations: int = 200,
    log_interval: int = 10,
    snapshot_iterations: list = [1, 50, 100, 150, 200],
    warmup_iterations: int = 10,
    model_id: str = "hubertsiuzdak/snac_24khz",
):
    """
    Profile memory stability under sustained decode load.

    NO empty_cache() calls - tests real-world memory behavior.

    Args:
        audio_seconds: Duration of audio to decode (15s default)
        total_iterations: Number of decode iterations (200 default)
        log_interval: Log memory stats every N iterations (10 default)
        snapshot_iterations: Save memory snapshots at these iterations
        warmup_iterations: Warmup iterations before tracking (10 default)
        model_id: HuggingFace model ID
    """
    import torch
    from snac import SNAC
    import os
    import math
    import pickle
    from collections import defaultdict

    os.environ["HF_HOME"] = "/cache/huggingface"

    print("=" * 70)
    print("SNAC MEMORY STABILITY PROFILER")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Audio duration: {audio_seconds}s")
    print(f"Total iterations: {total_iterations}")
    print(f"Snapshot iterations: {snapshot_iterations}")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model: {model_id}")
    model = SNAC.from_pretrained(model_id).cuda().eval()
    print(f"Model loaded. VQ strides: {model.vq_strides}")
    print()

    # Generate codes for fixed audio duration
    def generate_codes(audio_seconds):
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
            code = torch.randint(0, model.codebook_size, (1, code_len), device="cuda")
            codes.append(code)
        return codes

    codes = generate_codes(audio_seconds)
    print(f"Generated codes for {audio_seconds}s audio:")
    for i, c in enumerate(codes):
        print(f"  codes[{i}]: {tuple(c.shape)}")
    print()

    # Warmup phase (no tracking, no empty_cache)
    print(f"Warmup phase: {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model.decode(codes)
    torch.cuda.synchronize()
    print("Warmup complete.")
    print()

    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()

    # Tracking data structures
    memory_log = []
    snapshot_paths = []
    allocation_sizes_seen = set()

    # Main profiling loop
    print(f"Starting {total_iterations} decode iterations...")
    print("-" * 70)
    print(f"{'Iter':<8} {'Allocated(MB)':<15} {'Reserved(MB)':<15} {'MaxReserved(MB)':<18} {'NewAllocs':<10}")
    print("-" * 70)

    for iteration in range(1, total_iterations + 1):
        # Check if we should record memory history for this iteration
        should_snapshot = iteration in snapshot_iterations

        if should_snapshot:
            # Start recording memory history
            torch.cuda.memory._record_memory_history(
                # max_entries=500000, # todo actually find out how many entries does it record, hint : snapshot contains metadata
                stacks="python",
            )

        # Run decode
        with torch.no_grad():
            _ = model.decode(codes)
        torch.cuda.synchronize()

        if should_snapshot:
            # Save snapshot
            snapshot_path = f"/cache/stability_snapshot_iter_{iteration}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            snapshot_paths.append(snapshot_path)

            # Analyze snapshot for allocation sizes
            with open(snapshot_path, 'rb') as f:
                snapshot = pickle.load(f)

            new_sizes = set()
            if 'device_traces' in snapshot:
                for trace in snapshot.get('device_traces', []):
                    for event in trace:
                        if event.get('action') == 'alloc':
                            size = event.get('size', 0)
                            if size not in allocation_sizes_seen:
                                new_sizes.add(size)
                                allocation_sizes_seen.add(size)

        # Log memory stats at intervals
        if iteration % log_interval == 0 or iteration == 1:
            allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            max_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024

            new_allocs = len(new_sizes) if should_snapshot else "-"

            print(f"{iteration:<8} {allocated_mb:<15.2f} {reserved_mb:<15.2f} {max_reserved_mb:<18.2f} {new_allocs:<10}")

            memory_log.append({
                "iteration": iteration,
                "allocated_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "max_reserved_mb": max_reserved_mb,
            })

    print("-" * 70)
    print()

    # Analysis
    print("=" * 70)
    print("MEMORY STABILITY ANALYSIS")
    print("=" * 70)

    # Check for memory growth
    first_log = memory_log[0]
    last_log = memory_log[-1]

    allocated_growth = last_log["allocated_mb"] - first_log["allocated_mb"]
    reserved_growth = last_log["reserved_mb"] - first_log["reserved_mb"]

    print(f"\nMemory Growth (iteration 1 -> {total_iterations}):")
    print(f"  Allocated: {first_log['allocated_mb']:.2f} MB -> {last_log['allocated_mb']:.2f} MB (delta: {allocated_growth:+.2f} MB)")
    print(f"  Reserved:  {first_log['reserved_mb']:.2f} MB -> {last_log['reserved_mb']:.2f} MB (delta: {reserved_growth:+.2f} MB)")

    # Stability assessment
    print(f"\nStability Assessment:")
    if abs(allocated_growth) < 1.0:
        print(f"  [OK] Allocated memory stable (delta < 1MB)")
    else:
        print(f"  [WARN] Allocated memory growth detected: {allocated_growth:+.2f} MB")

    if abs(reserved_growth) < 10.0:
        print(f"  [OK] Reserved memory stable (delta < 10MB)")
    else:
        print(f"  [WARN] Reserved memory growth detected: {reserved_growth:+.2f} MB")

    # Check if max_reserved stabilized
    max_reserved_values = [log["max_reserved_mb"] for log in memory_log]
    if len(set(max_reserved_values[-5:])) == 1:
        print(f"  [OK] Max reserved memory stabilized at {max_reserved_values[-1]:.2f} MB")
    else:
        print(f"  [WARN] Max reserved memory still changing: {max_reserved_values[-5:]}")

    # Unique allocation sizes
    print(f"\nUnique allocation sizes observed: {len(allocation_sizes_seen)}")

    # Summary
    print(f"\n" + "=" * 70)
    print("SNAPSHOT FILES:")
    for path in snapshot_paths:
        print(f"  {path}")
    print("=" * 70)

    results = {
        "audio_seconds": audio_seconds,
        "total_iterations": total_iterations,
        "memory_log": memory_log,
        "snapshot_paths": snapshot_paths,
        "unique_allocation_sizes": len(allocation_sizes_seen),
        "allocated_growth_mb": allocated_growth,
        "reserved_growth_mb": reserved_growth,
        "final_allocated_mb": last_log["allocated_mb"],
        "final_reserved_mb": last_log["reserved_mb"],
        "max_reserved_mb": last_log["max_reserved_mb"],
    }

    return results


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": volume},
    timeout=3600,
)
def compare_snapshots(
    snapshot_paths: list = None,
):
    """
    Compare allocation patterns across snapshots to detect changes.
    """
    import pickle
    from collections import defaultdict

    if snapshot_paths is None:
        # Default paths
        snapshot_paths = [
            "/cache/stability_snapshot_iter_1.pickle",
            "/cache/stability_snapshot_iter_50.pickle",
            "/cache/stability_snapshot_iter_100.pickle",
            "/cache/stability_snapshot_iter_150.pickle",
            "/cache/stability_snapshot_iter_200.pickle",
        ]

    print("=" * 70)
    print("SNAPSHOT COMPARISON")
    print("=" * 70)

    for path in snapshot_paths:
        try:
            with open(path, 'rb') as f:
                snapshot = pickle.load(f)

            alloc_count = 0
            free_count = 0
            total_bytes = 0
            sizes = defaultdict(int)

            if 'device_traces' in snapshot:
                for trace in snapshot.get('device_traces', []):
                    for event in trace:
                        if event.get('action') == 'alloc':
                            alloc_count += 1
                            size = event.get('size', 0)
                            total_bytes += size
                            sizes[size] += 1
                        elif event.get('action') == 'free':
                            free_count += 1

            print(f"\n{path}:")
            print(f"  Allocations: {alloc_count}")
            print(f"  Frees: {free_count}")
            print(f"  Total allocated: {total_bytes / 1024 / 1024:.2f} MB")
            print(f"  Unique sizes: {len(sizes)}")

            # Top 5 allocation sizes
            top_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top allocation sizes (count):")
            for size, count in top_sizes:
                print(f"    {size / 1024:.1f} KB x {count}")

        except FileNotFoundError:
            print(f"\n{path}: NOT FOUND")

    return {"analyzed": len(snapshot_paths)}


@app.local_entrypoint()
def main(
    audio_seconds: float = 15.0,
    iterations: int = 200,
    compare: bool = False,
):
    """
    Run memory stability profiling.

    Args:
        audio_seconds: Duration of audio to decode
        iterations: Number of decode iterations
        compare: If True, compare existing snapshots instead of running new profile
    """
    if compare:
        results = compare_snapshots.remote()
        print("\nSnapshot comparison complete.")
    else:
        results = profile_memory_stability.remote(
            audio_seconds=audio_seconds,
            total_iterations=iterations,
        )
        print("\nMemory stability profiling complete.")
        print(f"\nSummary:")
        print(f"  Final allocated: {results['final_allocated_mb']:.2f} MB")
        print(f"  Final reserved: {results['final_reserved_mb']:.2f} MB")
        print(f"  Max reserved: {results['max_reserved_mb']:.2f} MB")
        print(f"  Allocated growth: {results['allocated_growth_mb']:+.2f} MB")
        print(f"  Reserved growth: {results['reserved_growth_mb']:+.2f} MB")
        print(f"  Unique allocation sizes: {results['unique_allocation_sizes']}")
