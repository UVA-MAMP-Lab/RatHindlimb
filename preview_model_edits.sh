#!/usr/bin/env bash
# Preview model_edits.qmd by executing and serving the HTML

set -e

# Colors
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}Rendering and starting preview...${NC}"

# Render the document
./render_model_edits.sh

# Start Quarto preview on the executed notebook (no re-execution needed)
echo -e "${GREEN}✓ Starting preview server...${NC}"
echo -e "${YELLOW}Note: Quarto will serve the already-executed notebook (no kernel needed)${NC}"
quarto preview model_edits_executed.ipynb
