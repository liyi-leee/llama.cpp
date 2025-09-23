#!/usr/bin/env python3
import numpy as np
from gguf import GGUFWriter, GGUFValueType
import sys
import os

def create_gguf_file(filename: str):
    """
    Creates a GGUF file with the same structure as the provided C++ code.
    """
    absolute_path = os.path.join(os.getcwd(), filename)
    arch = "llama"
    writer = GGUFWriter(filename, arch)

    print(f"Creating GGUF file '{filename}' with architecture '{arch}'")

    # --- Add Required Hyperparameters ---
    VOCAB_SIZE = 3
    BLOCK_COUNT = 1
    CONTEXT_LENGTH = 512
    EMBEDDING_LENGTH = 128
    FEED_FORWARD_LENGTH = 512
    HEAD_COUNT = 8
    HEAD_COUNT_KV = 1
    LAYER_NORM_RMS_EPS = 1e-5

    writer.add_string("general.name", "test-gguf")
    writer.add_uint32("llama.vocab_size", VOCAB_SIZE)
    writer.add_uint32("llama.block_count", BLOCK_COUNT)
    writer.add_uint32("llama.context_length", CONTEXT_LENGTH)
    writer.add_uint32("llama.embedding_length", EMBEDDING_LENGTH)
    writer.add_uint32("llama.feed_forward_length", FEED_FORWARD_LENGTH)
    writer.add_uint32("llama.attention.head_count", HEAD_COUNT)
    writer.add_uint32("llama.attention.head_count_kv", HEAD_COUNT_KV)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", LAYER_NORM_RMS_EPS)

    # --- Define Tokenizer (using specific methods and plain Python lists) ---
    tokens = ["<unk>", "<s>", "</s>"]
    scores = [0.0, 0.0, 0.0]
    toktypes = [2, 3, 3]

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    # --- Define and Add Tensors with Random Data ---
    print("Defining and adding model tensors with random data...")
    def add_random_tensor(name: str, shape: tuple[int, ...]):
        tensor = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
        writer.add_tensor(name, tensor)

    add_random_tensor("token_embd.weight", (VOCAB_SIZE, EMBEDDING_LENGTH))
    add_random_tensor("output.weight", (VOCAB_SIZE, EMBEDDING_LENGTH))
    add_random_tensor("output_norm.weight", (EMBEDDING_LENGTH,))
    add_random_tensor("blk.0.attn_norm.weight", (EMBEDDING_LENGTH,))
    add_random_tensor("blk.0.attn_q.weight", (EMBEDDING_LENGTH, EMBEDDING_LENGTH))
    add_random_tensor("blk.0.attn_k.weight", (EMBEDDING_LENGTH // HEAD_COUNT * HEAD_COUNT_KV, EMBEDDING_LENGTH ))
    add_random_tensor("blk.0.attn_v.weight", (EMBEDDING_LENGTH // HEAD_COUNT * HEAD_COUNT_KV, EMBEDDING_LENGTH ))
    add_random_tensor("blk.0.attn_output.weight", (EMBEDDING_LENGTH, EMBEDDING_LENGTH))
    add_random_tensor("blk.0.ffn_norm.weight", (EMBEDDING_LENGTH,))
    add_random_tensor("blk.0.ffn_gate.weight", (FEED_FORWARD_LENGTH, EMBEDDING_LENGTH, ))
    add_random_tensor("blk.0.ffn_down.weight", (EMBEDDING_LENGTH, FEED_FORWARD_LENGTH))
    add_random_tensor("blk.0.ffn_up.weight", (FEED_FORWARD_LENGTH, EMBEDDING_LENGTH))
    
    # --- Write GGUF File (Multi-step process based on the example) ---
    print(f"\nWriting file step-by-step to: {absolute_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close() 
    
    print(f"✅ GGUF test model written successfully.")
    
    # Final check to confirm file existence
    if os.path.exists(absolute_path):
        print(f"Confirmed: File '{filename}' now exists in the current directory.")
    else:
        print(f"Warning: File '{filename}' still not found after write operations.")


if __name__ == "__main__":
    create_gguf_file("test-gguf.gguf")