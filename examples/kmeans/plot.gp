set terminal png
set output 'plot.png'

plot '< paste pts.*.dat' with points lc rgb "black", \
     '< cat init_cluster_pts.1.dat' with points pt 5 lc rgb "green", \
     '< cat final_cluster_pts.1.dat' with points pt 5 lc rgb "red"
