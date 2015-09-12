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

write_nsec = []
read_nsec = []
num_shards = 0
for point, nsec in write_begins.items():
    write_start = nsec
    write_end = write_ends[point]
    read_start = write_end
    read_end = read_ends[point]
    point = point.replace(",", "-")

    num_shards += 1
    write_nsec.append(write_end - write_start);
    read_nsec.append(read_end - read_start);

write_sec = map(lambda ns: ns*1.0/10**9, write_nsec)
read_sec = map(lambda ns: ns*1.0/10**9, read_nsec)

min_write_sec = min(write_sec)
max_write_sec = max(write_sec)
avg_write_sec = sum(write_sec)/len(write_sec)
min_read_sec = min(read_sec)
max_read_sec = max(read_sec)
avg_read_sec = sum(read_sec)/len(read_sec)

print "num shards:", num_shards
print "write: min,max,avg:", min_write_sec, max_write_sec, avg_write_sec
print "read:  min,max,avg:", min_read_sec, max_read_sec, avg_read_sec
