#!/bin/bash

# Find the node.js binary
NODE_PATH="/nix/store/0akvkk9k1a7z5vjp34yz6dr91j776jhv-nodejs-20.11.1/bin/node"

if [ -x "$NODE_PATH" ]; then
  echo "Node.js found at $NODE_PATH"
  
  # Run the server
  exec "$NODE_PATH" server.js
else
  echo "Node.js not found at expected path"
  echo "Trying alternate locations..."
  
  for path in /nix/store/*/bin/node; do
    if [ -x "$path" ]; then
      echo "Found Node.js at $path"
      exec "$path" server.js
      exit 0
    fi
  done
  
  echo "Node.js not found in any location"
  exit 1
fi