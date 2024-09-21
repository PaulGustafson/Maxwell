# Maxwell's equations simulator

TODO:
1. Single GPU benchmarks
2. Optimize code for single GPU
3. Develop multi GPU following blog post below

References:
1. [Hackathon project proposal by Georgii Evtushenko](https://docs.google.com/document/d/1OxWw9aHeoUBFDOClcMr9UrPW8qmpdR5pPOcwH4jEhms/edit#heading=h.c3hqbft26ocn)
2. [Blog post on multi-GPU by Georgii](https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c)

## Steps to Reproduce Single GPU Benchmarking Results
- Benchmaarking script is called `fdtd_benchmark.cu`
- Compile via `nvcc -c main.cu -o main.o` and `nvcc fdtd_benchmark.cu main.o -o fdtd_benchmark`
- Run via `./fdtd_benchmark`
- Note: Results for Single GPU Benchmarking can be found [at this link](https://docs.google.com/spreadsheets/d/1krkTRtscSdfPweV9PB49tEk6Qup0lxjPkI4EdEInOlA/edit?gid=0#gid=0).
