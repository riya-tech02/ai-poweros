#!/bin/bash
# Complete deployment script for AI-PowerOS

set -e

echo "🚀 AI-PowerOS Deployment Script"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if API is running
check_api() {
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is running${NC}"
        return 0
    else
        echo -e "${RED}✗ API is not running${NC}"
        return 1
    fi
}

# Start services
start_services() {
    echo "Starting Docker services..."
    docker-compose up -d
    
    echo "Waiting for services to be ready (30s)..."
    sleep 30
    
    echo "Initializing databases..."
    python scripts/init_neo4j.py || echo "Neo4j init skipped"
    python scripts/create_kafka_topics.py || echo "Kafka init skipped"
    
    echo -e "${GREEN}✓ Services started${NC}"
}

# Stop services
stop_services() {
    echo "Stopping services..."
    docker-compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Run tests
run_tests() {
    echo "Running test suite..."
    python test_complete_system.py
}

# Show status
show_status() {
    echo ""
    echo "📊 System Status:"
    echo "================"
    
    # Check API
    if check_api; then
        curl -s http://localhost:8000/health | python -m json.tool
    fi
    
    echo ""
    echo "Docker Services:"
    docker-compose ps
    
    echo ""
    echo "🌐 Access Points:"
    echo "  • API Docs:    http://localhost:8000/docs"
    echo "  • Dashboard:   http://localhost:8000/dashboard/"
    echo "  • Neo4j:       http://localhost:7474"
    echo "  • Grafana:     http://localhost:3000"
    echo "  • Prometheus:  http://localhost:9090"
}

# Main menu
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    test)
        run_tests
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|test|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  test    - Run test suite"
        echo "  status  - Show system status"
        exit 1
        ;;
esac
