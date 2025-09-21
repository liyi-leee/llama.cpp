#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include "llama-batch.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml-impl.h"
#include "cgraph_generated.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>

#include <inttypes.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

// Helper function to convert ggml_type to TensorType
GGML::Serialization::TensorType convert_ggml_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return GGML::Serialization::TensorType_F32;
        case GGML_TYPE_F16: return GGML::Serialization::TensorType_F16;
        case GGML_TYPE_Q4_0: return GGML::Serialization::TensorType_Q4_0;
        case GGML_TYPE_Q4_1: return GGML::Serialization::TensorType_Q4_1;
        case GGML_TYPE_Q8_0: return GGML::Serialization::TensorType_Q8_0;
        case GGML_TYPE_I8: return GGML::Serialization::TensorType_I8;
        case GGML_TYPE_I16: return GGML::Serialization::TensorType_I16;
        case GGML_TYPE_I32: return GGML::Serialization::TensorType_I32;
        default: return GGML::Serialization::TensorType_F32;
    }
}

// Helper function to convert ggml_op to TensorOp
GGML::Serialization::TensorOp convert_ggml_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return GGML::Serialization::TensorOp_ADD;
        case GGML_OP_SUB: return GGML::Serialization::TensorOp_SUB;
        case GGML_OP_MUL: return GGML::Serialization::TensorOp_MUL;
        case GGML_OP_DIV: return GGML::Serialization::TensorOp_DIV;
        case GGML_OP_MUL_MAT: return GGML::Serialization::TensorOp_MUL_MAT;
        case GGML_OP_ROPE: return GGML::Serialization::TensorOp_ROPE;
        case GGML_OP_NORM: return GGML::Serialization::TensorOp_NORM;
        case GGML_OP_SOFT_MAX: return GGML::Serialization::TensorOp_SOFT_MAX;
        case GGML_OP_RESHAPE: return GGML::Serialization::TensorOp_RESHAPE;
        case GGML_OP_VIEW: return GGML::Serialization::TensorOp_VIEW;
        case GGML_OP_PERMUTE: return GGML::Serialization::TensorOp_PERMUTE;
        case GGML_OP_TRANSPOSE: return GGML::Serialization::TensorOp_TRANSPOSE;
        case GGML_OP_CONV_2D: return GGML::Serialization::TensorOp_CONV_2D;
        case GGML_OP_POOL_1D: return GGML::Serialization::TensorOp_POOL_1D;
        case GGML_OP_POOL_2D: return GGML::Serialization::TensorOp_POOL_2D;
        // For unary operations, we'll use specific cases when available
        case GGML_OP_UNARY: return GGML::Serialization::TensorOp_CUSTOM; // Handle unary ops specially
        default: return GGML::Serialization::TensorOp_CUSTOM;
    }
}

// 修改后的序列化函数，返回FlatBuffer元数据和独立的数据buffer
struct SerializationResult {
    std::unique_ptr<uint8_t[]> metadata_buffer;
    size_t metadata_size;
    std::unique_ptr<uint8_t[]> data_buffer;
    size_t data_size;
};

SerializationResult serialize_cgraph_to_flatbuffer(const struct ggml_cgraph * cgraph) {

    printf("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        printf(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                ggml_op_name(node->op), (node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, node) ? "g" : " ");
    }

    printf("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        printf(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                node->ne[0], node->ne[1],
                ggml_op_name(node->op),
                ggml_get_name(node));
    }

    flatbuffers::FlatBufferBuilder builder; // 使用默认大小，让它自动扩展

    // 创建一个映射来跟踪张量索引
    std::unordered_map<struct ggml_tensor*, uint32_t> tensor_index_map;
    
    // 收集所有唯一的张量 (nodes + leafs)
    std::vector<struct ggml_tensor*> all_tensors;
    
    // 添加所有叶子节点
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * tensor = cgraph->leafs[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }
    
    // 添加所有计算节点
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * tensor = cgraph->nodes[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }

    // 收集所有张量数据到独立的连续缓冲区（不受FlatBuffer限制）
    std::vector<uint8_t> continuous_data_buffer;
    std::vector<uint64_t> data_offsets; // 记录每个张量在缓冲区中的偏移
    
    size_t total_data_size = 0;
    
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * tensor = all_tensors[i];
        size_t data_size = ggml_nbytes(tensor);
        
        // 记录当前张量的数据偏移
        data_offsets.push_back(continuous_data_buffer.size());
        
        // 添加调试信息
        if (i < 5) {
            printf("Tensor %zu: data=%p, size=%zu, offset=%zu, name='%s'\n", 
                   i, tensor->data, data_size, continuous_data_buffer.size(), 
                   ggml_get_name(tensor) ? ggml_get_name(tensor) : "unnamed");
        }
        
        // 存储所有张量的数据到独立的buffer中
        if (tensor->data && data_size > 0) {
            const uint8_t* tensor_data = static_cast<const uint8_t*>(tensor->data);
            continuous_data_buffer.insert(continuous_data_buffer.end(), 
                                         tensor_data, tensor_data + data_size);
            total_data_size += data_size;
        } else {
            // 如果没有数据，添加占位符（零填充）
            continuous_data_buffer.resize(continuous_data_buffer.size() + data_size, 0);
            if (i < 5) {
                printf("  -> No data, zero-filled\n");
            }
        }
    }
    
    printf("Total data buffer size: %zu bytes\n", continuous_data_buffer.size());
    
    // 不再将数据存储在FlatBuffer中，FlatBuffer只存储元数据
    // 创建空的数据缓冲区向量（用于兼容性）
    std::vector<uint8_t> empty_data_buffer;
    auto data_buffer = builder.CreateVector(empty_data_buffer);
    
    // 创建序列化的张量
    std::vector<flatbuffers::Offset<GGML::Serialization::SerializedTensor>> tensor_offsets;
    
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * tensor = all_tensors[i];
        
        // 创建维度信息
        GGML::Serialization::TensorDimensions dims(
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]
        );
        
        // 创建步长信息
        GGML::Serialization::TensorStrides strides(
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]
        );
        
        // 创建操作参数（如果有的话）
        std::vector<int32_t> op_params_vec;
        // TODO: 根据具体操作类型填充参数
        auto op_params = builder.CreateVector(op_params_vec);
        
        // 创建源张量索引
        std::vector<uint32_t> src_indices;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (tensor->src[j] != nullptr) {
                auto it = tensor_index_map.find(tensor->src[j]);
                if (it != tensor_index_map.end()) {
                    src_indices.push_back(it->second);
                }
            }
        }
        auto src = builder.CreateVector(src_indices);
        
        // 创建名称
        const char* tensor_name = ggml_get_name(tensor);
        auto name = builder.CreateString(tensor_name ? tensor_name : "");
        
        // 计算数据大小
        size_t data_size = ggml_nbytes(tensor);
        
        // 创建序列化张量 - 数据偏移指向独立buffer中的位置
        auto serialized_tensor = CreateSerializedTensor(
            builder,
            convert_ggml_type(tensor->type),    // type
            &dims,                              // TensorDimensions
            &strides,                           // TensorStrides  
            convert_ggml_op(tensor->op),        // op
            op_params,                          // op_params
            tensor->flags,                      // flags
            src,                                // src
            0,                                  // view_src (TODO: handle view tensors)
            0,                                  // view_offs
            data_offsets[i],                    // data_offset (指向独立buffer)
            data_size,                          // data_size
            name                                // name
        );
        
        tensor_offsets.push_back(serialized_tensor);
    }
    
    auto tensors = builder.CreateVector(tensor_offsets);
    
    // 创建节点索引数组
    std::vector<uint32_t> node_indices_vec;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto it = tensor_index_map.find(cgraph->nodes[i]);
        if (it != tensor_index_map.end()) {
            node_indices_vec.push_back(it->second);
        }
    }
    auto node_indices = builder.CreateVector(node_indices_vec);
    
    // 创建叶子索引数组
    std::vector<uint32_t> leaf_indices_vec;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        auto it = tensor_index_map.find(cgraph->leafs[i]);
        if (it != tensor_index_map.end()) {
            leaf_indices_vec.push_back(it->second);
        }
    }
    auto leaf_indices = builder.CreateVector(leaf_indices_vec);
    
    // 创建梯度相关的索引数组（目前为空）
    auto grad_indices = builder.CreateVector<uint32_t>({});
    auto grad_acc_indices = builder.CreateVector<uint32_t>({});
    
    // 创建使用计数数组（目前为空）
    auto use_counts = builder.CreateVector<int32_t>({});

    // 创建HashSet（目前为空）
    auto visited_hash_set = GGML::Serialization::CreateHashSetDirect(
        builder,
        0,       // size
        nullptr, // used_bits
        nullptr  // keys
    );

    // 创建最终的序列化计算图
    auto serialized_cgraph = GGML::Serialization::CreateSerializedCGraph(
        builder,
        /* size */ (uint32_t)all_tensors.size(),
        /* n_nodes */ cgraph->n_nodes,
        /* n_leafs */ cgraph->n_leafs,
        tensors,
        node_indices,
        grad_indices,
        grad_acc_indices,
        leaf_indices,
        use_counts,
        visited_hash_set,
        GGML::Serialization::GraphEvalOrder_LEFT_TO_RIGHT,
        data_buffer
    );

    builder.Finish(serialized_cgraph);

    // 获取序列化数据
    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    printf("Serialized cgraph metadata: %p, size: %ld bytes\n", buf, size);
    printf("Total tensors: %zu, nodes: %d, leafs: %d\n", 
           all_tensors.size(), cgraph->n_nodes, cgraph->n_leafs);
    printf("Separate data buffer size: %zu bytes\n", continuous_data_buffer.size());
    
    // 复制FlatBuffer元数据到独立的内存块
    auto metadata_copy = std::make_unique<uint8_t[]>(size);
    memcpy(metadata_copy.get(), buf, size);
    
    // 复制数据buffer到独立的内存块
    auto data_copy = std::make_unique<uint8_t[]>(continuous_data_buffer.size());
    memcpy(data_copy.get(), continuous_data_buffer.data(), continuous_data_buffer.size());
    
    SerializationResult result;
    result.metadata_buffer = std::move(metadata_copy);
    result.metadata_size = size;
    result.data_buffer = std::move(data_copy);
    result.data_size = continuous_data_buffer.size();
    
    return result;
}

void convert_to_flat_cgraph(const struct ggml_cgraph * cgraph) {
    auto result = serialize_cgraph_to_flatbuffer(cgraph);
    // 这里可以将数据写入文件或进行其他处理
    printf("Serialization completed:\n");
    printf("  Metadata size: %zu bytes\n", result.metadata_size);
    printf("  Data size: %zu bytes\n", result.data_size);
}

// 测试函数：序列化然后直接验证 FlatBuffer 数据
void test_cgraph_serialization(const struct ggml_cgraph * original_cgraph) {
    printf("\n=== Testing CGraph Serialization ===\n");
    
    // 序列化
    printf("1. Serializing original cgraph...\n");
    auto result = serialize_cgraph_to_flatbuffer(original_cgraph);
    
    // 验证 FlatBuffer 数据
    printf("\n2. Verifying FlatBuffer metadata...\n");
    
    // 验证 FlatBuffer 数据的有效性
    auto verifier = flatbuffers::Verifier(result.metadata_buffer.get(), result.metadata_size);
    if (!GGML::Serialization::VerifySerializedCGraphBuffer(verifier)) {
        printf("✗ FlatBuffer metadata verification failed!\n");
        return;
    }
    printf("✓ FlatBuffer metadata is valid\n");
    
    // 获取序列化的计算图
    auto serialized_cgraph = GGML::Serialization::GetSerializedCGraph(result.metadata_buffer.get());
    if (!serialized_cgraph) {
        printf("✗ Failed to read SerializedCGraph from buffer!\n");
        return;
    }
    
    auto tensors = serialized_cgraph->tensors();
    if (!tensors) {
        printf("✗ No tensors found in serialized data!\n");
        return;
    }
    
    printf("✓ Successfully read serialized data\n");
    
    // 比较基本结构信息
    printf("\n3. Comparing structure information...\n");
    printf("Original: %d nodes, %d leafs\n", original_cgraph->n_nodes, original_cgraph->n_leafs);
    printf("Serialized: %d nodes, %d leafs\n", serialized_cgraph->n_nodes(), serialized_cgraph->n_leafs());
    
    bool structure_matches = (original_cgraph->n_nodes == serialized_cgraph->n_nodes()) &&
                            (original_cgraph->n_leafs == serialized_cgraph->n_leafs());
    
    if (!structure_matches) {
        printf("✗ Graph structure doesn't match!\n");
        return;
    }
    printf("✓ Graph structure matches\n");
    
    // 创建原始张量到索引的映射（与序列化时相同的逻辑）
    std::unordered_map<struct ggml_tensor*, uint32_t> tensor_index_map;
    std::vector<struct ggml_tensor*> all_tensors;
    
    // 添加所有叶子节点
    for (int i = 0; i < original_cgraph->n_leafs; i++) {
        struct ggml_tensor * tensor = original_cgraph->leafs[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }
    
    // 添加所有计算节点
    for (int i = 0; i < original_cgraph->n_nodes; i++) {
        struct ggml_tensor * tensor = original_cgraph->nodes[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }
    
    printf("\n4. Comparing tensor data...\n");
    printf("Total tensors to compare: %zu\n", all_tensors.size());
    
    if (tensors->size() != all_tensors.size()) {
        printf("✗ Tensor count mismatch: original=%zu, serialized=%u\n", 
               all_tensors.size(), tensors->size());
        return;
    }
    
    int mismatches = 0;
    int perfect_matches = 0;
    
    // 逐个比较张量
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * orig_tensor = all_tensors[i];
        auto serialized_tensor = tensors->Get(i);
        
        bool tensor_matches = true;
        
        // 比较维度
        auto dims = serialized_tensor->ne();
        if (orig_tensor->ne[0] != dims->ne0() || 
            orig_tensor->ne[1] != dims->ne1() ||
            orig_tensor->ne[2] != dims->ne2() || 
            orig_tensor->ne[3] != dims->ne3()) {
            tensor_matches = false;
            printf("  Tensor %zu: dimension mismatch\n", i);
            printf("    Original: [%ld,%ld,%ld,%ld]\n", 
                   orig_tensor->ne[0], orig_tensor->ne[1], orig_tensor->ne[2], orig_tensor->ne[3]);
            printf("    Serialized: [%ld,%ld,%ld,%ld]\n", 
                   dims->ne0(), dims->ne1(), dims->ne2(), dims->ne3());
        }
        
        // 比较步长
        auto strides = serialized_tensor->nb();
        if (orig_tensor->nb[0] != strides->nb0() || 
            orig_tensor->nb[1] != strides->nb1() ||
            orig_tensor->nb[2] != strides->nb2() || 
            orig_tensor->nb[3] != strides->nb3()) {
            tensor_matches = false;
            printf("  Tensor %zu: stride mismatch\n", i);
        }
        
        // 比较类型
        if (convert_ggml_type(orig_tensor->type) != serialized_tensor->type()) {
            tensor_matches = false;
            printf("  Tensor %zu: type mismatch\n", i);
        }
        
        // 比较操作
        if (convert_ggml_op(orig_tensor->op) != serialized_tensor->op()) {
            tensor_matches = false;
            printf("  Tensor %zu: operation mismatch (orig=%s, serialized=%d)\n", 
                   i, ggml_op_name(orig_tensor->op), serialized_tensor->op());
        }
        
        // 比较标志
        if (orig_tensor->flags != serialized_tensor->flags()) {
            tensor_matches = false;
            printf("  Tensor %zu: flags mismatch\n", i);
        }
        
        // 比较名称
        const char* orig_name = ggml_get_name(orig_tensor);
        const char* serialized_name = serialized_tensor->name() ? serialized_tensor->name()->c_str() : "";
        if (strcmp(orig_name ? orig_name : "", serialized_name) != 0) {
            tensor_matches = false;
            printf("  Tensor %zu: name mismatch (orig='%s', serialized='%s')\n", 
                   i, orig_name ? orig_name : "", serialized_name);
        }
        
        // 比较源张量索引
        auto src_indices = serialized_tensor->src();
        std::vector<uint32_t> expected_src_indices;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (orig_tensor->src[j] != nullptr) {
                auto it = tensor_index_map.find(orig_tensor->src[j]);
                if (it != tensor_index_map.end()) {
                    expected_src_indices.push_back(it->second);
                }
            }
        }
        
        if (src_indices->size() != expected_src_indices.size()) {
            tensor_matches = false;
            printf("  Tensor %zu: source count mismatch\n", i);
        } else {
            for (size_t j = 0; j < expected_src_indices.size(); j++) {
                if (src_indices->Get(j) != expected_src_indices[j]) {
                    tensor_matches = false;
                    printf("  Tensor %zu: source index mismatch at position %zu\n", i, j);
                    break;
                }
            }
        }
        
        // 比较数据大小
        size_t orig_data_size = ggml_nbytes(orig_tensor);
        
        if (orig_data_size != serialized_tensor->data_size()) {
            tensor_matches = false;
            printf("  Tensor %zu: data size mismatch (orig=%zu, serialized=%lu)\n", 
                   i, orig_data_size, serialized_tensor->data_size());
        }
        
        // 比较实际数据内容 - 使用独立的数据buffer
        if (orig_data_size > 0 && serialized_tensor->data_size() > 0) {
            if (serialized_tensor->data_offset() + serialized_tensor->data_size() <= result.data_size) {
                const uint8_t* serialized_data = result.data_buffer.get() + serialized_tensor->data_offset();
                const uint8_t* orig_data = static_cast<const uint8_t*>(orig_tensor->data);
                
                if (orig_data && orig_tensor->data) {
                    // 比较前64字节或全部数据（取较小值）
                    size_t compare_size = std::min(orig_data_size, (size_t)64);
                    bool data_matches = memcmp(orig_data, serialized_data, compare_size) == 0;
                    
                    if (!data_matches) {
                        tensor_matches = false;
                        printf("  Tensor %zu: data content mismatch (compared first %zu bytes)\n", i, compare_size);
                        
                        // 显示前几个字节的详细对比（仅对小张量）
                        if (orig_data_size <= 32 && i < 5) {
                            printf("    Original:   ");
                            for (size_t k = 0; k < std::min(compare_size, (size_t)16); k++) {
                                printf("%02x ", orig_data[k]);
                            }
                            printf("\n    Serialized: ");
                            for (size_t k = 0; k < std::min(compare_size, (size_t)16); k++) {
                                printf("%02x ", serialized_data[k]);
                            }
                            printf("\n");
                        }
                    } else if (i < 5) {
                        printf("  Tensor %zu: data content matches (checked %zu bytes)\n", i, compare_size);
                    }
                } else {
                    if (i < 5) {
                        printf("  Tensor %zu: original data is null, skipping data comparison\n", i);
                    }
                }
            } else {
                tensor_matches = false;
                printf("  Tensor %zu: invalid data buffer offset or size\n", i);
                if (i < 5) {
                    printf("    data_buffer=%p, offset=%llu, size=%llu, buffer_size=%zu\n",
                           result.data_buffer.get(), serialized_tensor->data_offset(), 
                           serialized_tensor->data_size(), result.data_size);
                }
            }
        } else if (i < 5) {
            printf("  Tensor %zu: no data to compare (orig_size=%zu, ser_size=%llu)\n", 
                   i, orig_data_size, serialized_tensor->data_size());
        }
        
        if (tensor_matches) {
            perfect_matches++;
        } else {
            mismatches++;
        }
        
        // 显示前几个张量的详细信息
        if (i < 5) {
            printf("  Tensor %zu (%s): %s\n", i, 
                   orig_name ? orig_name : "unnamed",
                   tensor_matches ? "✓ MATCH" : "✗ MISMATCH");
            printf("    Original: data=%p, size=%zu\n", orig_tensor->data, orig_data_size);
            printf("    Serialized: offset=%llu, size=%llu\n", 
                   serialized_tensor->data_offset(), serialized_tensor->data_size());
            if (!tensor_matches) {
                printf("    Mismatch details logged above\n");
            }
        } else if (i < 3) {
            printf("  Tensor %zu (%s): %s\n", i, 
                   orig_name ? orig_name : "unnamed",
                   tensor_matches ? "✓ MATCH" : "✗ MISMATCH");
        }
    }
    
    printf("\n5. Final comparison results:\n");
    printf("  Perfect matches: %d/%zu\n", perfect_matches, all_tensors.size());
    printf("  Mismatches: %d/%zu\n", mismatches, all_tensors.size());
    
    if (mismatches == 0) {
        printf("✓ All tensors match perfectly!\n");
        printf("✓ Serialization/deserialization test PASSED!\n");
    } else {
        printf("✗ Found %d tensor mismatches\n", mismatches);
        printf("✗ Serialization test FAILED!\n");
    }
    
    printf("=== Serialization test completed ===\n\n");
}

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);
    
    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(1, 1);

    //printf("%p\n", ctx->graph_reserve_get());

    // ggml_graph_print(ctx->graph_prev_get());

    // ggml_graph_print(ctx->graph_reserve_get());

    convert_to_flat_cgraph(ctx->graph_reserve_get());
    
    // 测试完整的序列化/反序列化流程
    test_cgraph_serialization(ctx->graph_reserve_get());

    auto * mem = llama_get_memory(ctx);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_memory_seq_rm(mem, -1, n_matching_session_tokens, -1);
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output is empty
                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            // if current token is not EOG, we add it to current assistant message
            if (params.conversation_mode && !waiting_for_first_input) {
                const auto id = common_sampler_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);

                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                if (buffer.empty()) { // Ctrl+D on empty line exits
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
