#!/bin/bash

#set -x

source $LIBMESH_DIR/examples/run_common.sh

example_name=particles_ex2

options="1 60 7.1 60 1 0"
run_example "$example_name" "$options"
