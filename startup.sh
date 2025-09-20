#!/bin/bash
# Startup script that runs tests and then starts the main application

set -e

echo "🚀 Embeddings Generator Container Starting..."
echo "=============================================="
echo ""
echo "📋 Startup Process:"
echo "   1. ✅ Container initialized"
echo "   2. 🧪 Running tests (this may take 10-30 seconds)..."
echo "   3. 🌐 Starting API server"
echo ""
echo "🧪 Running automated tests..."
echo "-----------------------------"

# Record start time
start_time=$(date +%s)

# Run the tests with more verbose output
echo "⏳ Test execution in progress..."
if pytest tests/ -v --tb=short --no-header; then
    # Calculate test duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "✅ All tests passed! (${duration}s)"
    echo "🌐 Starting Embeddings Generator API server..."
    echo "=============================================="
    echo ""
    echo "🔗 API will be available at:"
    echo "   • Main API: http://localhost:8000"
    echo "   • Documentation: http://localhost:8000/docs"
    echo "   • Health Check: http://localhost:8000/health"
    echo ""
    
    # Start the main application
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
else
    # Calculate test duration even on failure
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "❌ Tests failed after ${duration}s!"
    echo "=============================================="
    echo "🔍 Check the test output above for details."
    echo "💡 The application will not start due to test failures."
    echo "   This ensures only working code is deployed."
    echo ""
    exit 1
fi
