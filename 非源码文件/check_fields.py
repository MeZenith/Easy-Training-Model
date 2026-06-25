from gguf import GGUFReader

r = GGUFReader(r"workspace/exports/test_debug/model-F16.gguf")
print(f"Total fields: {len(r.fields)}")
for name, field in list(r.fields.items())[:30]:
    vtype_names = [t.name if hasattr(t, "name") else str(t) for t in field.types]
    parts_info = []
    for p in field.parts[:2]:
        parts_info.append(f"{type(p).__name__}({p.shape if hasattr(p,'shape') else '?'})")
    print(f"  {name}: parts={len(field.parts)} types={vtype_names} vals={','.join(parts_info)}")
