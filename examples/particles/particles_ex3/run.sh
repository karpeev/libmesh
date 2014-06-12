#!/bin/bash

#set -x

source $LIBMESH_DIR/examples/run_common.sh

example_name=particles_ex3

options="1 1 1 0"
run_example "$example_name" "$options"
