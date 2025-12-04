#!/bin/bash
# Force redeploy trigger - $(date)
pip install -r requirements.txt
echo "Build completed at $(date)"
