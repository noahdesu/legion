import sys
import re

record_re = re.compile("domain point: (?P<pt>.+);(?P<what>.+)seconds: "\
        "(?P<sec>\d+) nanos: (?P<nanosec>\d+)")

write_begins = {}
write_ends = {}
read_ends = {}

for line in sys.stdin.readlines():
    m = record_re.match(line)
    if not m:
        continue

    point = m.group("pt")
    nsec = long(m.group("sec")) * 10**9 + \
        long(m.group("nanosec"))

    what = m.group("what")
    if "write begins" in what:
        write_begins[point] = nsec
    elif "write ends" in what:
        write_ends[point] = nsec
    elif "read ends" in what:
        read_ends[point] = nsec
    else:
        assert False

for point, nsec in write_begins.items():
    write_start = nsec
    write_end = write_ends[point]
    read_start = write_end
    read_end = read_ends[point]

    point = point.replace(",", "-")
    print point, write_start, write_end, write_end-write_start,\
            read_start, read_end, read_end-read_start
