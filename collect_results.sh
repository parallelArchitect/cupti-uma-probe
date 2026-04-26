#!/bin/bash
# Package cupti_uma_probe results for sharing

cd "$(dirname "$0")"

if [ ! -f "cupti_uma_probe_results.json" ]; then
    echo "No results found. Run ./cupti_uma_probe first."
    exit 1
fi

OUTDIR="cupti_uma_results_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
cp cupti_uma_probe_results.json "$OUTDIR/"

zip -r "${OUTDIR}.zip" "$OUTDIR"
rm -rf "$OUTDIR"

echo ""
echo "Results packaged: $(pwd)/${OUTDIR}.zip"
echo ""
echo "What would you like to do?"
echo "  [1] Share — open GitHub Issues to upload"
echo "  [2] Local — keep results, no upload"
echo ""
read -p "Enter 1 or 2: " CHOICE

if [ "$CHOICE" = "1" ]; then
    xdg-open "https://github.com/parallelArchitect/cupti-uma-probe/issues/new" 2>/dev/null || \
    echo "Go to: https://github.com/parallelArchitect/cupti-uma-probe/issues/new"
else
    echo "Done."
fi
