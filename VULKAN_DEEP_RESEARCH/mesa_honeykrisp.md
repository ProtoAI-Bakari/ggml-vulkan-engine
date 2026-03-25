# Mesa Honeykrisp Vulkan Driver Analysis

## Sources
- asahilinux.org/2024/06/vk13-on-the-m1-in-1-month/
- Phoronix: Honeykrisp Sparse Support Mesa 25.1
- Mesa 26.0.0 Release Notes
- Mesa 25.2.0 Release Notes
- Alyssa Rosenzweig's blog posts

---

## What Honeykrisp IS

Honeykrisp is the Vulkan driver for Apple Silicon GPUs on Linux (Asahi Linux), part of the Mesa 3D graphics library. It is the **first conformant Vulkan implementation for Apple hardware on any OS** (Vulkan 1.3, 686,930 passing tests, 0 failures).

### Architecture

- **Based on**: NVK (NVIDIA's open-source Vulkan driver by Faith Ekstrand)
- **Adapted for**: Apple AGX GPU hardware (G13/G14/G15 architectures)
- **Compiler**: NIR (Mesa's intermediate representation) -> AGX ISA backend
- **Not from scratch**: Reuses significant infrastructure from NVK, especially for descriptor handling and command buffer management

---

## Vulkan Extensions Supported

### Core
- **Vulkan 1.3** (full conformance, no portability waivers)

### Extensions Added Over Time
- `VK_EXT_image_drm_format_modifier` - Zero-copy rendering
- `VK_EXT_custom_border_color` - D3D compatibility
- `VK_EXT_extended_dynamic_state` (1, 2, 3)
- `VK_EXT_vertex_input_dynamic_state`
- `VK_EXT_shader_object` - Pipelines optional
- `VK_KHR_present_id`, `VK_KHR_present_id2`
- `VK_KHR_present_wait`, `VK_KHR_present_wait2`
- `VK_KHR_pipeline_binary`
- `VK_EXT_shader_uniform_buffer_unsized_array`
- YCbCr support (software)
- Sparse resource support (Mesa 25.1)

### Extensions NOT Supported (Critical for Us)
- **VK_KHR_cooperative_matrix** - NOT SUPPORTED
- **VK_NV_cooperative_matrix2** - NOT SUPPORTED (NVIDIA-specific)
- VK_EXT_transform_feedback (planned)

---

## Shader Compilation Pipeline: NIR -> AGX ISA

### Overview

```
GLSL/SPIR-V source
    ↓
SPIR-V binary (offline via glslangValidator / shaderc)
    ↓
NIR (Mesa's intermediate representation)
    ↓ [nir_lower_* passes, optimization passes]
NIR (optimized)
    ↓ [AGX backend compiler]
AGX ISA binary (hardware instructions)
    ↓
Linked binary (prolog + body + epilog)
```

### Key Compilation Strategy

Honeykrisp divides hardware binaries into three parts:
1. **Prolog** - Setup state (vertex attributes, etc.)
2. **Shader body** - Main computation
3. **Epilog** - Output handling

Prologs and epilogs are compiled on-the-fly and cached. They handle dynamic state so the main shader body doesn't need recompilation for state changes.

### Dynamic State Strategies
1. **Conditional code** - Branch on state at runtime
2. **Precompiled variants** - Small set of common states
3. **Indirection** - Table lookups for state
4. **Prolog/epilog composition** - Concatenation or long jumps

### NIR Optimization Passes
Standard Mesa NIR passes apply, including:
- Algebraic simplification
- Dead code elimination
- Loop unrolling
- Vectorization (f16 pairs)
- Constant folding
- Register allocation (live range analysis)

---

## Known Performance Limitations

### 1. No Cooperative Matrix (THE Critical Limitation)

The AGX hardware has `simdgroup_matrix` instructions accessible via Metal. The Honeykrisp compiler does NOT expose these via `VK_KHR_cooperative_matrix`. This means:
- **All matmul must use manual FMA loops** in the shader
- **Prompt processing (GEMM) is ~6x slower** than Metal on the same hardware
- Token generation (GEMV) gap is ~2.8x (less affected since GEMV is bandwidth-bound)

### 2. Compiler Optimization Maturity

The AGX backend compiler is younger than Apple's Metal compiler (decades of refinement). Potential gaps:
- Instruction scheduling may not be optimal
- Register allocation might not be as aggressive
- Potentially missing hardware-specific peephole optimizations

### 3. Pipeline Barrier Overhead

The current Vulkan path inserts "a pipeline barrier between almost every dispatch" (per llama.cpp issue #10982). This creates synchronization overhead that Metal avoids through its command buffer model.

### 4. Descriptor Set Handling

Honeykrisp adapts NVK's descriptor set lowering for AGX hardware. The AGX descriptor model differs from NVIDIA's, so there's translation overhead. The driver "marries the set layout with hardware data structures."

---

## Compute-Specific Details

### Compute Shader Status
- Compute dispatch is **fully functional**
- Internal operations (buffer copies, image fills) use compute dispatch
- Query copies executed via compute shader emulation
- The driver can dispatch arbitrary compute work

### Draw Call Performance
- ~100 million draws/second (vkoverhead benchmark)
- This is primarily graphics, but indicates low overhead for command submission

### Buffer Copies
- Accelerated via compute shader dispatch with `VK_EXT_image_drm_format_modifier`
- Initially slow; zero-copy path significantly improved performance

---

## Status of VK_KHR_cooperative_matrix Support

### Current: NOT IMPLEMENTED

The extension requires:
1. **Hardware support**: AGX has `simdgroup_matrix` (8x8 MMA) -- the hardware capability exists
2. **Driver support**: Honeykrisp would need to:
   - Report cooperative matrix properties
   - Handle coopmat SPIR-V instructions
   - Lower to AGX `simdgroup_multiply_accumulate` instructions
3. **Compiler support**: NIR would need coopmat -> AGX ISA lowering

### Why It Doesn't Exist Yet

The Asahi GPU team has focused on:
1. Vulkan 1.3 conformance (achieved)
2. Gaming support (DXVK/vkd3d-proton)
3. AAA game support (October 2024)
4. Sparse resources (Mesa 25.1)
5. Pipeline binary support (Mesa 26.0)

Cooperative matrix for compute/ML has not been a priority compared to gaming features.

### What Would Be Needed

1. **NIR cooperative_matrix -> AGX lowering pass**: Map SPIR-V coopmat operations to AGX ISA's simdgroup_matrix instructions
2. **Property reporting**: Query and report supported matrix types (likely 8x8 f16 -> f32 accumulate)
3. **Testing**: Ensure correct results with llama.cpp's COOPMAT shader path
4. **Performance validation**: Verify that the coopmat path actually hits hardware acceleration

---

## How to Request Features / Contribute

### Filing Feature Requests

- **GitLab**: https://gitlab.freedesktop.org/mesa/mesa/-/issues
  - Tag with `asahi`, `honeykrisp` labels
  - Reference specific use case (llama.cpp ML inference)

### Contributing Code

- **Mesa development**: https://docs.mesa3d.org/developers.html
- The AGX backend is in `src/asahi/` within the Mesa tree
- The Honeykrisp driver is in `src/asahi/vulkan/`
- NIR passes are in `src/compiler/nir/`

### Key Contacts
- Alyssa Rosenzweig (lead Asahi GPU developer)
- Faith Ekstrand (NVK/Honeykrisp architecture)
- Asahi Linux community: #asahi on OFTC IRC, Matrix bridge

### Strategic Approach

The most impactful contribution would be:
1. Implement `VK_KHR_cooperative_matrix` for Honeykrisp
2. This would unlock the hardware MMA path in llama.cpp
3. Expected to close the ~3x gap in prompt processing
4. Also benefits all other Vulkan ML/compute workloads on Apple Silicon Linux

However, this is a significant engineering effort requiring:
- Deep understanding of both NIR and AGX ISA
- Access to Apple Silicon hardware for testing
- Coordination with Mesa maintainers
