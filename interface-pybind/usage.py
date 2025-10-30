# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn_op_runtime_predictor as torp
import json

#To use ttnn_op_runtime_predictor, run 'pip install .' in the ttnn-op-runtime-predictor root directory

#For docstrings, see interface-pybind/get_runtime_from_model_pybind.cpp or in python run 'help(ttnn_op_runtime_predictor)'

#Models take serialized json objects as input for objects such as ttnn tensors. 
#This is a sample json object for a ttnn tensor. The fields can be modified, and not every model 
#requires every field to be specified. See examples below. For specifics, refer to 
#ops/src/ops.cpp to see which fields are used for each op.
""" tensor_json = {
    "tensor_spec": {
        "logical_shape": dimensions,
        "tensor_layout": {
            "alignment": {"value": [1, 1, 32, 32]},
            "dtype": 0,
            "memory_config": {
                "buffer_type": 0,
                "created_with_nd_shard_spec": False,
                "memory_layout": 0
            },
            "page_config": {
                "config": {
                    "index": 1,
                    "value": {
                        "tile": {
                            "face_shape": [16, 16],
                            "num_faces": 4,
                            "tile_shape": [32, 32]
                        }
                    }
                }
            }
        }
    },
    "storage": {
        "index": 1,
        "value": {}
    }
} """
#To specify inputs such as datatypes and memory configs, use the following values:
#Dtype enum (from ops/include/ops.hpp):
""" typedef enum Dtype {

  BFLOAT16 = 0,
  FLOAT32 = 1,
  UINT32 = 2,
  BFLOAT8_B = 3,
  BFLOAT4_B = 4,
  UINT8 = 5,
  UINT16 = 6,
  INT32 = 7,
  INVALID = 8,

} DType; """

#Memory Config (Buffer type) enum:
""" typedef enum mem_cfg {
 
  DRAM = 0,
  L1 = 1,

} mem_cfg; """

#specifying some tensor dimensions for create_qkv_heads and concatenate_heads
num_heads = 32
num_kv_heads = 4
head_dim = 64
hidden_dim = (num_heads + 2 * num_kv_heads) * head_dim
batch_size = 1
seq_len = 512

#-----------------------------predict ttnn.experimental.create_qkv_heads runtime-----------------------------

#this is the expected json layout for create_qkv_heads input tensor
tensor_json = {
    "tensor_spec": {
        "logical_shape": [batch_size, 1, seq_len, hidden_dim],
        "tensor_layout": {
            "dtype": 0  #BF16
        }
    }
}

#query model
create_qkv_heads_runtime_prediction = torp.get_runtime_from_model(
    op_name="create_qkv_heads",
    tensor_json=tensor_json,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    transpose_k_heads=True,
)

print(f"Predicted runtime for create_qkv_heads: {create_qkv_heads_runtime_prediction}")

#-----------------------------predict ttnn.transformer.concatenate_heads runtime-----------------------------

#this is the expected json layout for concatenate_heads input tensor
tensor_json_2 = {
    "tensor_spec": {
        "logical_shape": [batch_size, num_heads, seq_len, head_dim],
        "tensor_layout": {
            "dtype": 0,  #BF16
            "memory_config": {
                "buffer_type": 0 #DRAM
            }
        }
    }
}

#specifying output memory config for concatenate_heads
output_memory_config = {
    "buffer_type": 0  #DRAM
}

#query model
concatenate_heads_runtime_prediction = torp.get_runtime_from_model(
    op_name="concatenate_heads",
    tensor_json=tensor_json_2,
    output_memory_config=output_memory_config
)

print(f"Predicted runtime for concatenate_heads: {concatenate_heads_runtime_prediction}")

#-----------------------------predict ttnn.exp runtime-----------------------------

#this is the expected json layout for exp input tensor
tensor_json_3 = {
    "tensor_spec": {
        "logical_shape": [1, 16, 256, 64],
        "tensor_layout": {
            "dtype": 0,  #BF16
            "memory_config": {
                "buffer_type": 0  #DRAM
            }
        }
    }
}

#query model
exp_runtime_prediction = torp.get_runtime_from_model(
    op_name="exp",
    tensor_json=tensor_json_3
)

print(f"Predicted runtime for exp: {exp_runtime_prediction}")

#-----------------------------predict ttnn.transformer.paged_scaled_dot_product_attention_decode runtime-----------------------------

#this is the expected json layout for paged_scaled_dot_product_attention_decode input tensors
q_tensor_json = {
    "tensor_spec": {
        "logical_shape": [1, 32, 1, 64],
        "tensor_layout": {
            "dtype": 0,  #BF16
            "memory_config": {
                "buffer_type": 0  #DRAM
            }
        }
    }
}
k_tensor_json = {
    "tensor_spec": {
        "logical_shape": [1, 4, 512, 64],
        "tensor_layout": {
            "dtype": 0,  #BF16
            "memory_config": {
                "buffer_type": 0  #DRAM
            }
        }
    }
}
v_tensor_json = {
    "tensor_spec": {
        "logical_shape": [1, 4, 512, 64],
        "tensor_layout": {
            "dtype": 0,  #BF16
            "memory_config": {
                "buffer_type": 0  #DRAM
            }
        }
    }
}
page_table_tensor_json = {
    "tensor_spec": {
        "logical_shape": [1, 1],
        "tensor_layout": {
            "dtype": 2,  #UINT32
            "memory_config": {
                "buffer_type": 0  #DRAM
            }
        }
    }
}

#specifying output memory config for paged_scaled_dot_product_attention_decode
output_memory_config_2 = {
    "buffer_type": 0  #DRAM
}

#query model
psdpa_decode_runtime_prediction = torp.get_runtime_from_model(
    op_name="paged_scaled_dot_product_attention_decode",
    q_tensor_json=q_tensor_json,
    k_tensor_json=k_tensor_json,
    v_tensor_json=v_tensor_json,
    page_table_tensor_json=page_table_tensor_json,
    optional_cur_pos_tensor_json=None,
    optional_attn_mask_tensor_json=None,
    is_causal=True,
    output_memory_config=output_memory_config_2,
    optional_scale=8.0,
    k_chunk_size=64,
    q_chunk_size=64,
    math_fidelity=2,
    math_approx_mode=1,
    fp32_dest_acc_en=1,
    packer_l1_acc=1,
    exp_approx_mode=1,
    use_sdpa_program_config=True,
    use_compute_kernel_config=True
)

print(f"Predicted runtime for paged_sdpa_decode: {psdpa_decode_runtime_prediction}")