#!/bin/bash
# Very simple environment diagnostic script

echo "BCBS Values Simple Diagnostics"
echo "============================"
echo ""

echo "Python executables:"
for python_path in /mnt/nixmodules/nix/store/*/bin/python*; do
  if [ -x "$python_path" ]; then
    echo "Found: $python_path"
    $python_path --version 2>&1
  fi
done

echo ""
echo "System information:"
uname -a

echo ""
echo "Environment variables:"
env | grep -E 'DATABASE|SESSION|API|PG' | sort

echo ""
echo "Directory contents:"
ls -la

echo ""
echo "Database check:"
if command -v psql &> /dev/null; then
  echo "psql command is available"
else 
  echo "psql command is not available"
fi

echo ""
echo "End of diagnostics"
echo "================="