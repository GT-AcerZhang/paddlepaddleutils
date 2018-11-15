#!/bin/bash
if [[ $# != 1 ]]; then
    echo "./debug.sh branch"
    exit 1
fi

./build.sh $1 release gpu
