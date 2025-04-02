#!/bin/bash
export PATH="$(pwd):$PATH"
exec ./python_workflow_wrapper.sh "$@"
