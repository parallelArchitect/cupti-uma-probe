/*
 * cupti_uma_probe.cu
 * CUPTI UVM Event Collection Test — GB10 Scope Diagnostic
 *
 * Tests the CUPTI UVM activity API directly — the same underlying layer
 * that Nsight Systems uses for UVM tracing. When Nsight UVM profiling is
 * unsupported on a platform, this tool determines whether the limitation
 * is at the CUPTI API level or above it.
 *
 * On discrete PCIe (Pascal, Turing, Ampere):
 *   CUPTI delivers UVM fault events on cudaMallocManaged access
 *
 * On GB10 hardware-coherent UMA:
 *   CUPTI may return success but deliver zero UVM events
 *   This is the known GB10 CUPTI scope limitation
 *
 * Build:
 *   nvcc -O2 -std=c++17 cupti_uma_probe.cu -o cupti_uma_probe \
 *        -lcudart -lcupti -I/usr/include
 *
 * Run:
 *   ./cupti_uma_probe
 *   ./cupti_uma_probe --json
 *
 * Author: parallelArchitect
 * Reference: https://forums.developer.nvidia.com/t/nsight-systems-unified-memory-trace-support-for-gb10-sm121/357848
 */

#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define BUFFER_SIZE     (8 * 1024 * 1024)   /* 8MB CUPTI activity buffer */
#define N_ELEMENTS      (1024 * 1024)        /* 4MB managed memory */
#define JSON_OUTPUT     "cupti_uma_probe_results.json"

/* ------------------------------------------------------------------ */
/* Error checking                                                       */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUPTI_CHECK(call) do { \
    CUptiResult e = (call); \
    if (e != CUPTI_SUCCESS) { \
        const char *msg; \
        cuptiGetResultString(e, &msg); \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", \
                __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

#define CUPTI_CHECK_SOFT(call, result) do { \
    (result) = (call); \
    if ((result) != CUPTI_SUCCESS) { \
        const char *msg; \
        cuptiGetResultString((result), &msg); \
        fprintf(stderr, "CUPTI soft error %s:%d: %s\n", \
                __FILE__, __LINE__, msg); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* State                                                                */
/* ------------------------------------------------------------------ */

static uint64_t g_uma_events       = 0;
static uint64_t g_cpu_fault_events = 0;
static uint64_t g_gpu_fault_events = 0;
static uint64_t g_total_records    = 0;

/* ------------------------------------------------------------------ */
/* CUPTI buffer callbacks                                              */
/* ------------------------------------------------------------------ */

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                       size_t *maxNumRecords) {
    *buffer = (uint8_t *)malloc(BUFFER_SIZE);
    if (!*buffer) { fprintf(stderr, "OOM allocating CUPTI buffer\n"); exit(1); }
    *size = BUFFER_SIZE;
    *maxNumRecords = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                       uint8_t *buffer, size_t size,
                                       size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize == 0) {
        free(buffer);
        return;
    }

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            g_total_records++;
            if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
                g_uma_events++;
                CUpti_ActivityUnifiedMemoryCounter2 *uma =
                    (CUpti_ActivityUnifiedMemoryCounter2 *)record;
                if (uma->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT)
                    g_cpu_fault_events++;
                if (uma->counterKind ==
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT)
                    g_gpu_fault_events++;
            }
        }
    } while (status == CUPTI_SUCCESS);

    free(buffer);
}

/* ------------------------------------------------------------------ */
/* GPU kernel — touches managed memory to trigger potential UVM events */
/* ------------------------------------------------------------------ */

__global__ void touch_kernel(float *data, uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] = data[tid] + 1.0f;
}

/* ------------------------------------------------------------------ */
/* Platform detection                                                   */
/* ------------------------------------------------------------------ */

static int is_hw_coherent_uma(int device) {
    int hpt = 0;
    cudaDeviceGetAttribute(&hpt,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
    int concurrent = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    concurrent = prop.concurrentManagedAccess;
    return hpt && concurrent;
}

/* ------------------------------------------------------------------ */
/* ISO timestamp                                                        */
/* ------------------------------------------------------------------ */

static void iso_ts(char *buf, size_t len) {
    time_t t = time(NULL);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    int json_mode = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--json") == 0) json_mode = 1;

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int hw_uma = is_hw_coherent_uma(device);

    if (!json_mode) {
        printf("=== CUPTI UVM Event Collection Probe ===\n");
        printf("GPU      : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("Platform : %s\n", hw_uma ? "HARDWARE_COHERENT_UMA" : "DISCRETE_PCIE");
        printf("Coherent : %s\n", hw_uma ? "yes (hardware)" : "no");
        printf("\n");
        printf("Testing CUPTI UVM activity event collection...\n\n");
    }

    /* --- Step 1: Allocate managed memory to load UVM driver --- */
    float *data = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, N_ELEMENTS * sizeof(float)));

    if (!json_mode)
        printf("[1] cudaMallocManaged: SUCCESS (%zu MB)\n",
               (N_ELEMENTS * sizeof(float)) / (1024*1024));

    /* --- Step 2: CPU touch to activate UVM driver --- */
    if (!json_mode) { printf("[2] CPU touch (activate UVM)... "); fflush(stdout); }
    for (size_t i = 0; i < N_ELEMENTS; i++) data[i] = (float)i;
    if (!json_mode) printf("done\n");

    /* --- Step 3: GPU touch to activate UVM on GPU side --- */
    if (!json_mode) { printf("[3] GPU touch (activate UVM GPU side)... "); fflush(stdout); }
    int blocks = (N_ELEMENTS + 255) / 256;
    touch_kernel<<<blocks, 256>>>(data, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!json_mode) printf("done\n\n");

    /* --- Step 4: NOW register CUPTI callbacks (UVM driver active) --- */
    CUptiResult cupti_result;
    CUPTI_CHECK_SOFT(cuptiActivityRegisterCallbacks(buffer_requested,
                                                    buffer_completed),
                     cupti_result);

    int callbacks_registered = (cupti_result == CUPTI_SUCCESS);

    if (!json_mode)
        printf("[4] cuptiActivityRegisterCallbacks: %s\n",
               callbacks_registered ? "SUCCESS" : "FAILED");

    /* --- Step 5: Enable UVM counter activity (UVM driver now loaded) --- */
    CUptiResult enable_result;
    CUPTI_CHECK_SOFT(cuptiActivityEnable(
        CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER), enable_result);

    int uma_enabled = (enable_result == CUPTI_SUCCESS);
    const char *enable_msg;
    cuptiGetResultString(enable_result, &enable_msg);

    if (!json_mode)
        printf("[5] cuptiActivityEnable(UNIFIED_MEMORY_COUNTER): %s (%s)\n",
               uma_enabled ? "SUCCESS" : "FAILED", enable_msg);

    /* --- Step 6: CPU touch again — CUPTI now active, should capture events --- */
    if (!json_mode) { printf("[6] CPU touch (CUPTI active)... "); fflush(stdout); }
    float sum = 0;
    for (size_t i = 0; i < N_ELEMENTS; i++) { data[i] = 0; sum += data[i]; }
    (void)sum;
    if (!json_mode) printf("done\n");

    /* --- Step 7: GPU touch again — CUPTI active --- */
    if (!json_mode) { printf("[7] GPU touch (CUPTI active)... "); fflush(stdout); }
    touch_kernel<<<blocks, 256>>>(data, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!json_mode) printf("done\n");

    /* --- Step 8: CPU re-touch after GPU --- */
    if (!json_mode) { printf("[8] CPU re-touch after GPU... "); fflush(stdout); }
    sum = 0;
    for (size_t i = 0; i < N_ELEMENTS; i++) sum += data[i];
    (void)sum;
    if (!json_mode) printf("done\n");

    /* --- Step 9: Flush CUPTI buffers --- */
    CUDA_CHECK(cudaDeviceSynchronize());
    cuptiActivityFlushAll(1);

    if (!json_mode) {
        printf("\n=== Results ===\n");
        printf("Total CUPTI records received : %lu\n", g_total_records);
        printf("UVM activity events          : %lu\n", g_uma_events);
        printf("  CPU page fault events      : %lu\n", g_cpu_fault_events);
        printf("  GPU page fault events      : %lu\n", g_gpu_fault_events);
        printf("\n");

        if (!uma_enabled) {
            printf("STATUS: CUPTI UVM activity could not be enabled\n");
            printf("        Return code: %s\n", enable_msg);
        } else if (g_uma_events == 0) {
            printf("STATUS: CUPTI UVM activity enabled but ZERO events received\n");
            if (hw_uma)
                printf("        This is consistent with the known GB10 CUPTI scope\n");
            else
                printf("        Unexpected on discrete PCIe — investigation needed\n");
        } else {
            printf("STATUS: CUPTI UVM events received normally\n");
            printf("        Platform behaves as expected\n");
        }
        printf("\nJSON : %s\n", JSON_OUTPUT);
    }

    /* --- Write JSON report --- */
    char ts[64]; iso_ts(ts, sizeof(ts));
    FILE *f = fopen(JSON_OUTPUT, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"tool\": \"cupti-uma-probe\",\n");
        fprintf(f, "  \"version\": \"1.0.0\",\n");
        fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
        fprintf(f, "  \"platform\": {\n");
        fprintf(f, "    \"gpu_name\": \"%s\",\n", prop.name);
        fprintf(f, "    \"sm_major\": %d,\n", prop.major);
        fprintf(f, "    \"sm_minor\": %d,\n", prop.minor);
        fprintf(f, "    \"hw_coherent_uma\": %s\n", hw_uma ? "true" : "false");
        fprintf(f, "  },\n");
        fprintf(f, "  \"cupti\": {\n");
        fprintf(f, "    \"callbacks_registered\": %s,\n",
                callbacks_registered ? "true" : "false");
        fprintf(f, "    \"uma_activity_enabled\": %s,\n",
                uma_enabled ? "true" : "false");
        fprintf(f, "    \"uma_enable_result\": \"%s\",\n", enable_msg);
        fprintf(f, "    \"total_records\": %lu,\n", g_total_records);
        fprintf(f, "    \"uma_events\": %lu,\n", g_uma_events);
        fprintf(f, "    \"cpu_fault_events\": %lu,\n", g_cpu_fault_events);
        fprintf(f, "    \"gpu_fault_events\": %lu\n", g_gpu_fault_events);
        fprintf(f, "  },\n");
        fprintf(f, "  \"status\": \"%s\"\n",
                !uma_enabled ? "CUPTI_UMA_ENABLE_FAILED" :
                g_uma_events == 0 ? "CUPTI_UMA_EVENTS_ZERO" :
                "CUPTI_UMA_EVENTS_OK");
        fprintf(f, "}\n");
        fclose(f);
    }

    CUDA_CHECK(cudaFree(data));
    return 0;
}
