# cupti-uma-probe

CUPTI UVM Activity API diagnostic for GB10 / DGX Spark.

## Background

CUPTI (CUDA Profiling Tools Interface) is NVIDIA's C-based interface
for building profiling and tracing tools. Nsight Systems uses CUPTI
internally via the Activity API to collect UVM traces.

This tool calls the CUPTI Activity API directly — bypassing Nsight's
toolchain — to determine whether CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
events are supported at the API level on GB10.

When Nsight UVM profiling is unsupported on a platform, this tool
helps determine whether the limitation exists at the CUPTI Activity
API level or above it.

Nsight Systems UVM profiling confirmed unsupported on GB10:
https://forums.developer.nvidia.com/t/nsight-systems-unified-memory-trace-support-for-gb10-sm121/357848

---

## What It Tests

UVM driver must be active before CUPTI can subscribe to UVM events.
The tool loads UVM first, then enables CUPTI — this is the required
initialization order.

1. cudaMallocManaged — loads UVM driver
2. CPU touch — activates UVM
3. GPU touch — activates UVM on the GPU side
4. cuptiActivityRegisterCallbacks — registers buffer callbacks
5. cuptiActivityEnable(UNIFIED_MEMORY_COUNTER) — enables UVM events
6. CPU + GPU access with CUPTI active — should trigger events
7. Reports CUPTI return codes and event counts

## Expected Results

| Platform | Expected |
|----------|----------|
| Discrete PCIe (Pascal through Ampere) | UVM events received |
| GB10 hardware-coherent UMA | Under investigation |

---

## Build

GB10 systems with both CUDA 13.0 and 13.1 installed: use 13.0.
CUDA 13.1 has a known event timing issue on GB10.

```bash
# Default
nvcc -O2 -std=c++17 cupti_uma_probe.cu -o cupti_uma_probe \
     -lcudart -lcupti

# If cupti.h not found, specify include path explicitly
nvcc -O2 -std=c++17 cupti_uma_probe.cu -o cupti_uma_probe \
     -lcudart -lcupti \
     -I$(CUDA_HOME)/extras/CUPTI/include
```

## Run

```bash
./cupti_uma_probe          # human-readable output + JSON log
./cupti_uma_probe --json   # JSON only
```

---

## Share Results

If you have a GB10 / DGX Spark, share your JSON output via
GitHub Issues — this tool is actively collecting GB10 results:
https://github.com/parallelArchitect/cupti-uma-probe/issues

---

## Related Tools

nvidia-uma-fault-probe (PTX-based latency, bandwidth, coherence):
https://github.com/parallelArchitect/nvidia-uma-fault-probe

sparkview (continuous system health monitor):
https://github.com/parallelArchitect/sparkview

GB10 hardware baseline findings:
https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851

---

## Author

parallelArchitect — Human-directed GPU engineering with AI assistance.

## License

MIT
