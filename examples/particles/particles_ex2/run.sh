#!/bin/bash

#set -x

source $LIBMESH_DIR/examples/run_common.sh

example_name=particles_ex2

options=""
run_example "$example_name" "$options"
