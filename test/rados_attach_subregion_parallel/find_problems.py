BASE_RANGE = 65536
PARTS_RANGE = 1024
for base in range(1,BASE_RANGE):
    nelements = base ** 2
    for regions in range(1,PARTS_RANGE):
        sr = regions ** 0.5
        if sr != int(sr):
            continue
        sr = nelements * 1.0 / regions
        sr = sr ** 0.5
        if sr != int(sr):
            continue
        part_size_mb = nelements * 8 / regions / 2**20
        if part_size_mb < 16:
            continue
        total_mb = nelements * 8 / 2**20
        print nelements, base, regions, total_mb, part_size_mb
