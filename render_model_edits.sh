#!/usr/bin/env bash
# Wrapper script to render model_edits.qmd with graceful OpenSim cleanup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Rendering model_edits.qmd...${NC}"

# Set up environment
export PATH="/home/hudson/RatHindlimb/.venv/bin:$PATH"
cd /home/hudson/RatHindlimb

# Step 1: Convert QMD to IPYNB
echo -e "${YELLOW}Step 1: Converting QMD to notebook...${NC}"
quarto convert model_edits.qmd

# Step 2: Execute notebook with graceful cleanup
echo -e "${YELLOW}Step 2: Executing notebook...${NC}"
python execute_notebook.py

# Step 3: Render to HTML without execution
echo -e "${YELLOW}Step 3: Rendering to HTML...${NC}"
quarto render model_edits_executed.ipynb --to html

echo -e "${GREEN}✓ Successfully rendered to model_edits_executed.html${NC}"

# Optional: Open in browser if --preview flag is passed
if [ "$1" == "--preview" ]; then
    echo -e "${YELLOW}Opening in browser...${NC}"
    if command -v xdg-open &> /dev/null; then
        xdg-open model_edits_executed.html
    elif command -v open &> /dev/null; then
        open model_edits_executed.html
    else
        echo -e "${YELLOW}Please open model_edits_executed.html in your browser${NC}"
    fi
fi
