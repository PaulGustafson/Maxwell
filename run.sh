#!/bin/bash
rm -rf build  && mkdir -p build/data && cd build && cmake .. && make && ./cuda_project 100 100 && cd ..
