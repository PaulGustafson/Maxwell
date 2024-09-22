#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <string>

constexpr float dt = 1.0f;
constexpr float initial_t = 1.0f;
constexpr float dx = 1.0f;
constexpr float dy = 1.0f;
constexpr float C0_p_dt = 1.0f;
constexpr int source_position = 10;


class NvtxScope {
 public:
    explicit NvtxScope(const char *scope_name) { push_nvtx_range(scope_name); }
    explicit NvtxScope(const std::string &scope_name) : NvtxScope(scope_name.c_str()) {}

    ~NvtxScope() { nvtxRangePop(); }

 private:
    void push_nvtx_range(const char *name) {
        nvtxEventAttributes_t event_attributes = {0};
        event_attributes.version = NVTX_VERSION;
        event_attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

        // Configure the Attributes
        event_attributes.colorType = NVTX_COLOR_ARGB;
        event_attributes.color = 0xFF6FDFFF; 
        event_attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event_attributes.message.ascii = name;

        nvtxRangePushEx(&event_attributes);
    }
};

void baseline_fdtd_cg(
const int n_steps,
// param to scale problem size
const int nx, const int ny, const int n_cells, 
float *  ez, 
const float * mh,
// ez and mh update hx and hy
float * hx, float * hy,
// hx and hy update dz
float * dz,
// dz and er update ez
const float * er
);