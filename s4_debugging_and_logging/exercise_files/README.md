# cProfile
\# Command to use cProfile \
\# <sort_order> ncalls, tottime, percall, cumtime, percall \
$ python -m cProfile -o <output_file> -s <sort_order> myscript.py \

# Pytorch profiler
\# Check pytorch_profiler to know how to save log \
\# To launch dashboard and open the http://localhost:6006/#pytorch_profiler \
$ tensorboard --logdir=./log
