"""GGUF 量化：F16 → Q8_0 / Q4_K_M"""
import sys, os
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize


def quantize_gguf(input_path: str, output_path: str, quant_type: str = "Q8_0"):
    print(f"Reading: {input_path}")
    reader = GGUFReader(input_path)

    print(f"Quantizing to {quant_type}...")

    arch_bytes = reader.get_field("general.architecture").parts[-1]
    arch = bytes(arch_bytes).decode("utf-8") if isinstance(arch_bytes, (bytes, bytearray)) else "qwen2"

    writer = GGUFWriter(output_path, arch)

    # Copy metadata
    for field in reader.fields.values():
        for i, val in enumerate(field.parts):
            vtype = field.types[i] if i < len(field.types) else field.types[0]
            writer.add_key_value(field.name, val, vtype)

    # Copy tensors with quantization
    total = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        data = tensor.data
        shape = tensor.shape

        if name.endswith("weight") and len(data.shape) >= 2:
            try:
                q_type = getattr(__import__('gguf.quants', fromlist=[quant_type]), quant_type)
                q_data = quantize(data, q_type)
                writer.add_tensor(name, q_data, raw_shape=shape)
            except Exception as e:
                print(f"  Quant failed for {name}: {e}, keeping F16")
                writer.add_tensor(name, data, raw_shape=shape)
        else:
            writer.add_tensor(name, data, raw_shape=shape)

        if i % 50 == 0:
            print(f"  {i}/{total} {name}")

    print("Writing output...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python quantize_gguf.py <input.gguf> <output.gguf> <Q8_0|Q4_K_M>")
        sys.exit(1)
    quantize_gguf(sys.argv[1], sys.argv[2], sys.argv[3])
