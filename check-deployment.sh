#!/bin/bash

echo "🔍 InterVA Docker Deployment Checklist"
echo "======================================"
echo ""

# Check if files exist
echo "✓ Checking required files..."
files=("Dockerfile" "requirements.txt" ".dockerignore" "render.yaml" "main.py" "index.html")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file MISSING!"
    fi
done

echo ""
echo "✓ Checking vman3 module..."
if [ -d "vman3" ]; then
    echo "  ✓ vman3/ directory exists"
else
    echo "  ✗ vman3/ directory MISSING!"
fi

echo ""
echo "✓ Checking data files..."
if [ -f "probbase2022.csv" ] || [ -f "vman3/interva/data/probbase2022.csv" ]; then
    echo "  ✓ probbase2022.csv found"
else
    echo "  ⚠ probbase2022.csv not found (may cause issues)"
fi

echo ""
echo "======================================"
echo "🐳 Ready to build Docker image!"
echo ""
echo "Next steps:"
echo "1. Test locally:"
echo "   docker build -t interva-api ."
echo "   docker run -p 8000:8000 interva-api"
echo ""
echo "2. Deploy to Render.com:"
echo "   - Push to GitHub"
echo "   - Connect repository in Render dashboard"
echo "   - Render will auto-deploy using render.yaml"
echo ""
echo "See DEPLOYMENT.md for detailed instructions."
