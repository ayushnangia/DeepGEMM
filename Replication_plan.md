
# Detailed Plan: Porting DeepGEMM from CUDA/Hopper to ROCm/HIP for AMD MI300X (gfx942)

**Goal:** Create a high-performance FP8 GEMM library functionally equivalent to DeepGEMM, optimized for the AMD MI300X (gfx942 architecture) using the ROCm/HIP ecosystem. This involves a near-complete rewrite of performance-critical components.

**Target Audience:** Developers with deep expertise in AMD GPU architecture (CDNA3), HIP, MFMA intrinsics, LDS optimization, and the original DeepGEMM codebase.

**Prerequisites:**

*   **Hardware:** AMD MI300X GPU(s).
*   **Software:**
    *   Compatible Linux Distribution (refer to AMD ROCm documentation).
    *   ROCm Toolkit (e.g., ROCm 5.x or 6.x supporting gfx942). Ensure installation includes:
        *   HIP SDK (`hipcc`, runtime libraries)
        *   ROCm Developer Tools (`rocminfo`, `rocProf`, `rocgdb`)
    *   Python environment (e.g., 3.8+) with PyTorch ROCm build.
    *   Optional: Containerized environment (Docker) with ROCm for reproducibility.
*   **Verification:** Confirm environment using `rocminfo` (should list gfx942), `hipconfig --full`, and compiling/running basic HIP examples.

---

**Phase 1: Environment Setup & Foundational Porting (Setup & Host Code)**

1.  **ROCm Environment Setup & Validation:**
    *   Install drivers and ROCm toolkit following AMD's official documentation for the target OS and MI300X.
    *   Verify installation thoroughly (see Prerequisites).
    *   Install PyTorch with ROCm support compatible with the installed ROCm version.
2.  **Project Fork & Branching:**
    *   Fork the DeepGEMM repository.
    *   Create a dedicated branch (e.g., `amd-mi300x-port`) for this effort. Use version control diligently.
3.  **Initial CUDA-to-HIP Conversion (Host Code Focus):**
    *   **Target Files:** Python scripts (`setup.py`, `*.py` in `deep_gemm/jit`, `deep_gemm/jit_kernels`, `tests`), `CMakeLists.txt` (for IDE), non-kernel C++ utilities (if any, though most are in headers).
    *   **Tooling:** Apply `hipify-perl` or `hipify-clang` for an initial pass.
    *   **Manual Review & Correction:**
        *   Replace `cuda...` API calls with `hip...` equivalents (e.g., `cudaMalloc` -> `hipMalloc`, `cudaMemcpy` -> `hipMemcpy`, `cudaStream_t` -> `hipStream_t`).
        *   Change namespaces (`cuda::std` -> `std`, check other `cuda::` usage).
        *   Update header includes (`cuda_runtime.h` -> `hip/hip_runtime.h`, `cuda_fp8.h` -> check ROCm FP8 support or use custom definitions, `cuda_bf16.h` -> `hip/hip_bfloat16.h`).
        *   Address potential signature mismatches or missing HIP equivalents for less common CUDA APIs.
    *   **Kernel Files (`*.cuh`):** Mark these files clearly (e.g., rename to `.hiph.template` or similar) – they need a full manual rewrite, not just hipification.
4.  **Build System Adaptation (`setup.py`, `jit/compiler.py`):**
    *   **Compiler Detection:** Implement `get_hipcc_compiler()` in `jit/compiler.py` to find `/opt/rocm/bin/hipcc` or use `ROCM_PATH`.
    *   **Flags:** Update `build()` in `jit/compiler.py` to use `hipcc` flags:
        *   Architecture: `--offload-arch=gfx942`
        *   Optimization: `-O3`
        *   Shared Object: `-shared -fPIC`
        *   C++/HIP Standard: `-std=c++17` (or as needed)
        *   Includes: `-I/opt/rocm/include` etc.
        *   Math libs: `-lm` potentially needed.
    *   **Dependencies:** Remove logic related to CUTLASS/CuTe includes in `setup.py`. Decide if ROCm library headers (e.g., rocWMMA) need to be packaged or assumed present in the environment. For now, rely on standard HIP headers.
    *   **Caching:** Modify the cache signature hashing in `jit/compiler.py` to include ROCm version, `hipcc` path/flags, and gfx942 target.
5.  **Remove/Replace NVIDIA-Specific Code:**
    *   **Delete:** `deep_gemm/jit/interleave_ffma.py`. Remove its import and call from `jit/compiler.py`.
    *   **Remove:** `cuobjdump` calls, references to PTX/SASS, `__nv_fp8*`, `__nv_bfloat16` (use `hip_bfloat16`, define custom FP8 or use ROCm equivalents if available), `CUtensorMap`, TMA functions/constants, WGMMA types/functions, `cutlass::` references, NVIDIA-specific intrinsics/pragmas (e.g., `__shfl_sync` -> `__shfl`).
    *   **CUDA_HOME:** Remove dependencies on `CUDA_HOME`.

---

**Phase 2: Core Kernel Rewrite (Device Code - HIP/MFMA/LDS)**

*   **Goal:** Reimplement kernel logic using HIP primitives optimized for CDNA3 (gfx942).

1.  **MMA Logic (WGMMA -> MFMA Intrinsics):**
    *   **Architecture Study:** Understand MI300X MFMA details: supported input/output types (FP8, BF16, FP16, FP32), available instruction shapes (e.g., `mfma_f32_32x32x8f8_e4m3`, `mfma_f32_16x16x16f8_e4m3` - verify exact opcodes for gfx942), and register requirements (AGPRs/VGPRs).
    *   **Intrinsic Header:** `#include <hip/amd_detail/amd_mfma.h>`.
    *   **Implementation (`mfma_utils.hiph`):** Create wrappers or directly use MFMA intrinsics within the main kernel loop over the K-dimension.
    *   **Data Flow:** Design the flow: Load A/B tiles from LDS to VGPRs -> Execute sequence of MFMA instructions -> Accumulate results in FP32 VGPRs.
    *   **Register Allocation:** Carefully manage VGPR usage, as MFMA instructions consume many registers. This impacts occupancy.
    *   **FP8/BF16 Types:** Use `_Float8` (if supported by hipcc) or custom structs for FP8. Use `hip_bfloat16`. Ensure correct type casting for MFMA inputs.
    *   **Scaling:** Apply LHS/RHS scaling factors *after* the MFMA accumulation (during promotion to FP32), similar to the original design. Fetch scales efficiently.
2.  **Memory Operations (TMA -> Async Copy / Vector Ops / LDS):**
    *   **Global Memory Addressing:** Calculate global memory source/destination addresses for tiles based on `hipBlockIdx_x`, `hipBlockIdx_y`, `hipThreadIdx_x`, tensor dimensions, and strides directly in the kernel. Remove all `CUtensorMap` logic.
    *   **Global -> LDS:**
        *   **Option A (Async Copy):** Use `hipMemcpyAsync` with the kernel's `hipStream_t` to copy tiles from global memory to LDS buffers. Requires careful synchronization (`__syncthreads`).
        *   **Option B (Vector Loads):** For potentially higher bandwidth, explore using GCN ISA vector load instructions (e.g., `buffer_load_dwordx4`, `flat_load_dwordx4`) via inline assembly (`asm volatile(...)`) or compiler builtins if available. Requires intricate address calculations, alignment considerations, and bounds checking. Higher complexity, higher potential reward.
    *   **LDS -> Global:**
        *   Use `hipMemcpyAsync` or GCN ISA vector stores (`buffer_store_dwordx4`, `flat_store_dwordx4`) from LDS result buffers back to global memory.
    *   **Pipeline (`fp8_gemm.hiph`):** Rebuild the K-dimension software pipeline (`kNumStages`): Use multiple LDS buffers for A and B tiles. Coordinate async copies into buffer `s+1` while computing on buffer `s`. Use `__syncthreads()` to ensure data is available in LDS before compute starts and compute is finished before overwriting LDS buffers.
3.  **Shared Memory Adaptation (SMEM -> LDS):**
    *   **LDS Layout Design:** This is critical for MFMA and vector op performance.
        *   Design LDS tile layouts for A and B matrices that facilitate efficient loading into VGPRs for MFMA instructions and minimize LDS bank conflicts. This might involve padding within the tile or using swizzled layouts (e.g., 64x64 threads accessing a tile).
        *   Analyze CDNA3 LDS bank structure (32 banks, typical access size). Avoid patterns where multiple threads in a wavefront access the same bank simultaneously.
    *   **Size Calculation (`get_lds_config`):** Create a HIP equivalent function in Python (`jit_kernels/utils.py`) to calculate total LDS required based on block dimensions, MFMA shapes, pipeline depth (`kNumStages`), data types (FP8, BF16, FP32 accumulators if spilled), and LDS padding/layout overhead. Ensure this respects the MI300X LDS size limit per CU / per workgroup (e.g., 64KB per CU often shared).
    *   **Remove NVIDIA Utils:** Eliminate `get_swizzle_mode`, `get_block_n_padding_for_smem_d`.
4.  **Synchronization:**
    *   **Block Level:** Replace `cutlass::arch::*Barrier` and kernel-wide `__syncthreads()` with `__syncthreads()`. Use it to synchronize between pipeline stages (load complete, compute complete), and before/after critical LDS read/writes.
    *   **Wavefront Level:** Replace `__syncwarp()` usages. Often, `__syncthreads()` is sufficient. If fine-grained intra-wavefront sync is truly needed (rare in GEMMs), investigate AMD GCN synchronization primitives or memory fences (`__threadfence_block()`).
5.  **Kernel Structure (`fp8_gemm.hiph`, `scheduler.hiph`):**
    *   **Host Launcher:** Define a C++ host function (e.g., `launch_fp8_gemm_kernel`) that takes dimensions, pointers, stream, grid/block sizes, and LDS size, then calls `hipLaunchKernelGGL`. This will be called by the JIT `launch` function.
    *   **Templating:** Update kernel templates (`fp8_gemm.hiph.template`?) to accept HIP types, MFMA configurations, LDS layout parameters, pipeline depth, etc.
    *   **Scheduler:** Re-evaluate block rasterization/swizzling (`get_swizzled_block_idx`). Test if the original pattern improves L1/L2 cache reuse on MI300X or if a simpler linear scheduler is better initially. Adapt `get_global_idx` for HIP kernel argument conventions. The scheduler logic for Grouped Masked needs careful porting to handle group lookups (`__ldg` replacement might need host pre-processing or careful global memory access).

---

**Phase 3: JIT System Adaptation (Python Integration)**

1.  **Compiler Integration (`jit/compiler.py`):**
    *   Complete `get_hipcc_compiler()` and `build()` function modifications to correctly invoke `hipcc` with all necessary flags (`--offload-arch=gfx942`, includes, optimization, shared lib flags). Ensure error handling for compilation failures.
    *   Refine cache key generation to robustly capture the ROCm/HIP build environment.
2.  **Code Generation (`jit/template.py`):**
    *   Finalize HIP header includes (`hip/hip_runtime.h`, `hip/hip_bfloat16.h`, potentially custom FP8/MFMA headers).
    *   Verify type mappings (`gencode_map`, `ctype_map`, `map_ctype`) for `hipStream_t`, `hip_bfloat16`, FP8 representations, and pointers work correctly with `ctypes` and the HIP C ABI.
    *   Ensure the generated C++ template code correctly calls the *new* HIP host launcher function with arguments cast appropriately.
3.  **Kernel Launch Logic (`jit_kernels/*.py`):**
    *   **C++ Template Strings:** Update the embedded C++ code strings in `gemm.py`, `m_grouped_gemm.py` to match the rewritten HIP kernel structure, includes, types, and host launcher function signature. Parameter names (`{BLOCK_M}`, etc.) need to align with the new HIP template.
    *   **`get_best_configs` Rewrite:** This requires significant effort.
        *   Gather MI300X architectural data (CU count, LDS size/banks, registers/CU, MFMA throughputs, cache sizes/bandwidth, HBM bandwidth).
        *   Define MI300X-specific search spaces for tunable parameters (block dims, pipeline stages, maybe LDS layout strategies).
        *   Develop new heuristics based on maximizing occupancy (waves per CU), leveraging MFMA throughput, minimizing LDS bank conflicts, and balancing compute with memory bandwidth, all tailored to gfx942. **Expect this to be highly iterative and likely require feedback from Phase 5 tuning.** An initial guess might be block sizes like 128x128 or 256x128 (workgroup size 256 threads often good), pipeline depth 3-5.
    *   **LDS Calculation:** Implement `get_lds_config` (renamed from `get_smem_config`) accurately.
    *   **Argument Preparation:** Replace `get_col_major_tma_aligned_tensor` with `prepare_lhs_scales` (or similar) if the chosen LDS layout or MFMA input requirements for scales necessitate specific transpositions or padding (different from TMA's 16-byte alignment).

---

**Phase 4: Testing & Validation**

1.  **Adapt JIT Tests (`tests/test_jit.py`):**
    *   Test finding `hipcc`.
    *   Test compiling a minimal HIP kernel (e.g., simple vector add or just argument printing).
    *   Verify `ctypes` interaction with HIP types (`hipStream_t`, `hip_bfloat16`, etc.).
2.  **Adapt Core Tests (`tests/test_core.py`):**
    *   **Reference:** Ensure PyTorch ROCm build is functioning correctly on MI300X. Compare `torch.matmul`/`torch.einsum` results with known good values or rocBLAS outputs. Note any base precision differences.
    *   **FP8:** If using custom FP8 handling, write unit tests for the conversion functions against standard definitions.
    *   **Correctness Execution:** Run `test_gemm`, `test_m_grouped_gemm_contiguous`, `test_m_grouped_gemm_masked` against the ported HIP kernels.
    *   **Debugging:** Use `rocgdb` for debugging kernel crashes or hangs. Use `printf` sparingly from HIP kernels for value inspection (compile with flags allowing printf). Compare intermediate results (e.g., values in LDS after load, accumulated values before store) with a simplified reference kernel if necessary.
    *   **Grouped Layouts:** Pay special attention to the `grouped_layout` pointer usage in the HIP kernel for contiguous/masked modes. Test edge cases (empty groups, single group, many groups).

---

**Phase 5: Performance Analysis & Tuning**

1.  **Profiling (`rocProf`):**
    *   Learn `rocProf` usage for MI300X: collecting kernel timings, hardware counters (occupancy, cache hits, LDS bank conflicts, memory R/W, MFMA busy %).
    *   Profile the ported kernels across various shapes from `test_core.py`.
2.  **Benchmarking:**
    *   Use `rocProf` for accurate kernel times or adapt `bench_kineto` to use `hipEvent`-based timing, ensuring proper stream synchronization and warm-up.
    *   Ensure stable GPU clocks (e.g., set power limits or clocks using ROCm SMI tools if needed) for consistent measurements.
3.  **Iterative Tuning Loop:**
    *   Analyze profiling data to find bottlenecks (e.g., low occupancy -> check register/LDS usage; low MFMA utilization -> check memory latency hiding/pipeline depth; memory bound -> improve global memory access/LDS reuse; LDS conflicts -> change LDS layout/padding).
    *   Modify `get_best_configs` heuristics or implement a simple auto-tuner within `JITTuner` to explore different block sizes, pipeline depths, LDS layouts.
    *   Re-compile (via JIT), re-benchmark, re-profile. Repeat.
    *   **Advanced:** If needed, inspect the GCN ISA generated by `hipcc` (`--save-temps` or via `llvm-objdump`) to understand low-level code generation and identify optimization opportunities missed by the compiler.
4.  **Comparison:** Benchmark against `rocBLAS` FP8 GEMM functions for the same shapes and data types. Aim to be competitive, especially for shapes relevant to the original DeepGEMM targets.

---

**Phase 6: Documentation & Cleanup**

1.  **README Update:** Add detailed sections for ROCm/MI300X: Installation (ROCm version, PyTorch ROCm), Build Instructions, Usage Notes (any functional differences?), Performance expectations/results.
2.  **Code Documentation:** Add extensive comments in `.hiph` files explaining MFMA logic, LDS layout choices, synchronization points, and pipeline structure. Document Python heuristics in `get_best_configs`.
3.  **Code Cleanup:** Remove all vestiges of CUDA/NVIDIA code. Ensure consistent HIP API usage. Add HIP error checking (`HIP_CHECK` macro) around relevant API calls in C++ host code (if any) and potentially within the JIT launch wrapper.

---

**Potential Risks & Challenges:**

*   **ROCm Ecosystem Maturity:** ROCm/HIP, especially for cutting-edge hardware and features like FP8, may be less mature than CUDA, potentially leading to compiler bugs, performance inconsistencies, or less comprehensive documentation/tooling.
*   **Performance Parity:** Achieving the same performance level as the highly optimized Hopper version will be extremely challenging due to fundamental architectural differences and NVIDIA's mature tooling/libraries (CUTLASS).
*   **Low-Level Complexity:** Optimizing effectively requires deep dives into MFMA intrinsics, GCN ISA, LDS banking, and memory hierarchies, which has a steep learning curve.
*   **FP8 Handling:** Consistent and performant FP8 support across the ROCm stack (compiler intrinsics, libraries, PyTorch) needs verification.
*   **Debugging:** Debugging complex GPU kernels, especially those involving async operations and intricate memory layouts, remains difficult.

This detailed plan provides a structured approach but acknowledges the significant technical depth and iterative effort required for a successful port.
