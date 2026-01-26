"""
SNAC Decoder Profiler

Custom profiling wrapper combining CUDA event timers with PyTorch Profiler
for identifying optimization targets in the SNAC decoder.

Usage:
    python snac_profiler.py [--device cuda] [--batch-size 1] [--audio-seconds 5.0]
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import sys


@dataclass
class TimingResult:
    """Stores timing results for a single operation."""
    name: str
    cuda_time_ms: float
    count: int = 1

    @property
    def avg_ms(self) -> float:
        return self.cuda_time_ms / self.count


@dataclass
class ProfileResult:
    """Complete profiling results."""
    timings: Dict[str, TimingResult] = field(default_factory=dict)
    total_decode_ms: float = 0.0
    peak_memory_mb: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SNAC Decoder Profiling Results",
            "=" * 60,
            f"{'Operation':<35} {'Time (ms)':<12} {'% Total':<10}",
            "-" * 60,
        ]

        sorted_timings = sorted(
            self.timings.values(),
            key=lambda x: x.cuda_time_ms,
            reverse=True
        )

        for t in sorted_timings:
            pct = (t.cuda_time_ms / self.total_decode_ms * 100) if self.total_decode_ms > 0 else 0
            lines.append(f"{t.name:<35} {t.cuda_time_ms:>10.3f}  {pct:>8.1f}%")

        lines.extend([
            "-" * 60,
            f"{'Total decode time':<35} {self.total_decode_ms:>10.3f}",
            f"{'Peak GPU memory (MB)':<35} {self.peak_memory_mb:>10.1f}",
            "=" * 60,
        ])

        return "\n".join(lines)


class CUDATimer:
    """Context manager for timing CUDA operations using events."""

    def __init__(self, name: str, results: Dict[str, TimingResult]):
        self.name = name
        self.results = results
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = self.start_event.elapsed_time(self.end_event)

        if self.name in self.results:
            self.results[self.name].cuda_time_ms += elapsed_ms
            self.results[self.name].count += 1
        else:
            self.results[self.name] = TimingResult(self.name, elapsed_ms)


class SNACProfiler:
    """
    Profiler for SNAC decoder that provides:
    - Per-operation CUDA timing
    - PyTorch Profiler integration for kernel-level analysis
    - Memory tracking
    - Chrome trace export
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.results = ProfileResult()
        self._timings: Dict[str, TimingResult] = {}

    def _reset_timings(self):
        self._timings = {}
        torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def _time_op(self, name: str):
        """Context manager for timing individual operations."""
        yield CUDATimer(name, self._timings)

    def generate_test_codes(
        self,
        batch_size: int = 1,
        audio_seconds: float = 5.0
    ) -> List[torch.Tensor]:
        """
        Generate realistic test codes matching SNAC output structure.

        SNAC 24kHz:
        - hop_length = 3*3*7*7 = 441
        - At 24kHz, 1 second = 24000 samples = ~54 latent frames
        - vq_strides = [8, 4, 2, 1] means:
          - codes[0]: T/8 tokens (coarsest)
          - codes[1]: T/4 tokens
          - codes[2]: T/2 tokens
          - codes[3]: T tokens (finest)
        """
        # Calculate base sequence length
        samples = int(audio_seconds * self.model.sampling_rate)
        # Pad to alignment
        hop = self.model.hop_length
        attn_window = self.model.attn_window_size or 1
        import math
        lcm = math.lcm(self.model.vq_strides[0], attn_window)
        pad_to = hop * lcm
        padded_samples = math.ceil(samples / pad_to) * pad_to

        # Latent sequence length
        latent_len = padded_samples // hop

        # Generate codes for each VQ level
        codes = []
        for stride in self.model.vq_strides:
            code_len = latent_len // stride
            code = torch.randint(
                0, self.model.codebook_size,
                (batch_size, code_len),
                device=self.device
            )
            codes.append(code)

        return codes

    def profile_decode_detailed(
        self,
        codes: List[torch.Tensor],
        warmup: int = 10,
        iterations: int = 50
    ) -> ProfileResult:
        """
        Profile decode() with detailed per-operation timing.

        Instruments:
        - VQ from_codes (per codebook)
        - Decoder attention
        - Decoder blocks (per stage)
        - Snake activations
        """
        self._reset_timings()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model.decode(codes)

        torch.cuda.synchronize()

        # Timed iterations with CUDA events
        for _ in range(iterations):
            with torch.no_grad():
                # Time total decode
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                # === VQ from_codes ===
                vq_start = torch.cuda.Event(enable_timing=True)
                vq_end = torch.cuda.Event(enable_timing=True)
                vq_start.record()
                z_q = self.model.quantizer.from_codes(codes)
                vq_end.record()

                # === Decoder forward ===
                dec_start = torch.cuda.Event(enable_timing=True)
                dec_end = torch.cuda.Event(enable_timing=True)
                dec_start.record()
                audio = self.model.decoder(z_q)
                dec_end.record()

                end.record()
                torch.cuda.synchronize()

                # Record timings
                total_ms = start.elapsed_time(end)
                vq_ms = vq_start.elapsed_time(vq_end)
                dec_ms = dec_start.elapsed_time(dec_end)

                self._add_timing("total_decode", total_ms)
                self._add_timing("vq_from_codes", vq_ms)
                self._add_timing("decoder_forward", dec_ms)

        # Compute averages
        for name, timing in self._timings.items():
            timing.cuda_time_ms /= iterations
            timing.count = 1

        self.results.timings = self._timings.copy()
        self.results.total_decode_ms = self._timings["total_decode"].cuda_time_ms
        self.results.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return self.results

    def _add_timing(self, name: str, ms: float):
        if name in self._timings:
            self._timings[name].cuda_time_ms += ms
            self._timings[name].count += 1
        else:
            self._timings[name] = TimingResult(name, ms)

    def profile_with_pytorch_profiler(
        self,
        codes: List[torch.Tensor],
        warmup: int = 5,
        active: int = 5,
        export_trace: Optional[str] = None
    ) -> str:
        """
        Profile using PyTorch's built-in profiler for kernel-level analysis.

        Returns table of top CUDA operations and optionally exports Chrome trace.
        """
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model.decode(codes)

        torch.cuda.synchronize()

        # Profile
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(active):
                with torch.no_grad():
                    with record_function("snac_decode"):
                        _ = self.model.decode(codes)
                torch.cuda.synchronize()

        # Export Chrome trace if requested
        if export_trace:
            prof.export_chrome_trace(export_trace)
            print(f"Chrome trace exported to: {export_trace}")

        # Return table sorted by CUDA time
        return prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30
        )

    def benchmark_latency(
        self,
        codes: List[torch.Tensor],
        warmup: int = 50,
        iterations: int = 200
    ) -> Dict[str, float]:
        """
        Benchmark decode latency with statistics.

        Returns dict with mean, std, p50, p95, p99 latencies.
        """
        import numpy as np

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model.decode(codes)

        torch.cuda.synchronize()

        # Collect timing samples
        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = self.model.decode(codes)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = np.array(times)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

    def profile_snake_activations(
        self,
        codes: List[torch.Tensor],
        iterations: int = 20
    ) -> Dict[str, float]:
        """
        Profile Snake1d activations specifically to quantify their impact.

        Counts and times all Snake1d calls in the decoder.
        """
        from snac.layers import Snake1d

        snake_times = []
        snake_count = 0

        # Hook to time Snake1d forward passes
        def snake_hook(module, input, output):
            nonlocal snake_count
            snake_count += 1

        # Register hooks
        hooks = []
        for name, module in self.model.decoder.named_modules():
            if isinstance(module, Snake1d):
                hooks.append(module.register_forward_hook(snake_hook))

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = self.model.decode(codes)

        # Count Snake calls
        snake_count = 0
        with torch.no_grad():
            _ = self.model.decode(codes)
        snake_calls_per_decode = snake_count

        # Remove hooks
        for h in hooks:
            h.remove()

        return {
            "snake_calls_per_decode": snake_calls_per_decode,
            "snake_modules_in_decoder": len(hooks),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile SNAC decoder")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--audio-seconds", type=float, default=5.0, help="Audio duration")
    parser.add_argument("--export-trace", type=str, default=None, help="Export Chrome trace path")
    parser.add_argument("--model", default="hubertsiuzdak/snac_24khz", help="Model to load")
    args = parser.parse_args()

    print(f"Loading SNAC model from {args.model}...")

    # Add snac-decoder to path
    sys.path.insert(0, "/home/andy/flash-snac/snac-decoder")
    from snac import SNAC

    model = SNAC.from_pretrained(args.model)

    print(f"Model loaded. Device: {args.device}")
    print(f"Config: sampling_rate={model.sampling_rate}, decoder_dim={model.decoder_dim}")
    print(f"VQ strides: {model.vq_strides}, codebook_size={model.codebook_size}")
    print()

    profiler = SNACProfiler(model, device=args.device)

    # Generate test codes
    print(f"Generating test codes for batch_size={args.batch_size}, audio={args.audio_seconds}s...")
    codes = profiler.generate_test_codes(args.batch_size, args.audio_seconds)
    for i, c in enumerate(codes):
        print(f"  codes[{i}]: shape={tuple(c.shape)}")
    print()

    # Profile Snake activations
    print("Profiling Snake1d activation count...")
    snake_info = profiler.profile_snake_activations(codes)
    print(f"  Snake1d modules in decoder: {snake_info['snake_modules_in_decoder']}")
    print(f"  Snake1d calls per decode: {snake_info['snake_calls_per_decode']}")
    print()

    # Detailed profiling
    print("Running detailed profiling (50 iterations)...")
    results = profiler.profile_decode_detailed(codes, warmup=10, iterations=50)
    print(results.summary())
    print()

    # Benchmark latency
    print("Running latency benchmark (200 iterations)...")
    latency = profiler.benchmark_latency(codes, warmup=50, iterations=200)
    print(f"Latency statistics:")
    print(f"  Mean:  {latency['mean_ms']:.3f} ms")
    print(f"  Std:   {latency['std_ms']:.3f} ms")
    print(f"  P50:   {latency['p50_ms']:.3f} ms")
    print(f"  P95:   {latency['p95_ms']:.3f} ms")
    print(f"  P99:   {latency['p99_ms']:.3f} ms")
    print()

    # Real-time factor
    audio_ms = args.audio_seconds * 1000
    rtf = audio_ms / latency['mean_ms']
    print(f"Real-time factor: {rtf:.1f}x (>{1.0:.1f}x means faster than real-time)")
    print()

    # PyTorch profiler
    print("Running PyTorch profiler for kernel-level analysis...")
    trace_path = args.export_trace or "snac_trace.json"
    table = profiler.profile_with_pytorch_profiler(codes, export_trace=trace_path)
    print(table)


if __name__ == "__main__":
    main()
