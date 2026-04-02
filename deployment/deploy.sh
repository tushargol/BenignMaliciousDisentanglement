#!/bin/bash

# Power Systems Industrial IDS Deployment Script
# This script deploys the complete Power Systems IDS using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$DEPLOYMENT_DIR")"
COMPOSE_FILE="$DEPLOYMENT_DIR/docker-compose.yml"

# Power Systems Configuration
POWER_SYSTEMS_MODE=${POWER_SYSTEMS_MODE:-true}
NERC_CIP_MODE=${NERC_CIP_MODE:-true}
SUBSTATION_ID=${SUBSTATION_ID:-SUB001}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] POWER SYSTEMS IDS: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking Power Systems IDS prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if we can access Docker
    if ! docker info &> /dev/null; then
        error "Cannot access Docker. Please check Docker permissions."
    fi
    
    # SECURITY: Check for environment file
    if [ ! -f "$DEPLOYMENT_DIR/.env" ]; then
        warn "Environment file .env not found."
        warn "Please copy .env.template to .env and configure your credentials."
        warn "Template location: $DEPLOYMENT_DIR/.env.template"
        
        # Check if template exists
        if [ ! -f "$DEPLOYMENT_DIR/.env.template" ]; then
            error "Environment template .env.template not found. Cannot proceed."
        fi
        
        # Ask user if they want to create .env from template
        read -p "Would you like to create .env from template? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$DEPLOYMENT_DIR/.env.template" "$DEPLOYMENT_DIR/.env"
            warn "Created .env file from template."
            warn "IMPORTANT: Please edit .env and replace placeholder values before deploying!"
            warn "Press Enter to continue after configuring .env, or Ctrl+C to abort."
            read -r
        else
            error "Environment file required for deployment. Please configure .env file first."
        fi
    fi
    
    # SECURITY: Check for default/placeholder values in .env
    if grep -q "your-.*-here\|example-.*-change-in-production\|CHANGE.*DEFAULT" "$DEPLOYMENT_DIR/.env"; then
        error "Default or placeholder values detected in .env file. Please configure all credentials before deploying."
    fi
    
    # Check if Power Systems models exist
    if [ ! -f "$PROJECT_DIR/outputs/models/power_autoencoder.pt" ] && [ ! -f "$PROJECT_DIR/outputs/models/autoencoder.pt" ]; then
        error "Power Systems autoencoder model not found. Please run training first."
    fi
    
    if [ ! -f "$PROJECT_DIR/outputs/models/power_classifier.pt" ] && [ ! -f "$PROJECT_DIR/outputs/models/classifier.pt" ]; then
        error "Power Systems classifier model not found. Please run training first."
    fi
    
    # Check Power Systems environment variables
    if [ -z "$SUBSTATION_ID" ]; then
        warn "SUBSTATION_ID not set, using default: SUB001"
        export SUBSTATION_ID="SUB001"
    fi
    
    # SECURITY: Verify environment file permissions
    if [ -f "$DEPLOYMENT_DIR/.env" ]; then
        env_perms=$(stat -c "%a" "$DEPLOYMENT_DIR/.env" 2>/dev/null || stat -f "%A" "$DEPLOYMENT_DIR/.env" 2>/dev/null)
        if [[ "$env_perms" =~ [456][0-9][0-9] ]]; then
            warn "Environment file has permissive permissions ($env_perms). Consider chmod 600 .env"
        fi
    fi
    
    log "Power Systems prerequisites check passed."
    log "Substation ID: $SUBSTATION_ID"
    log "NERC CIP Mode: $NERC_CIP_MODE"
    log "Security checks completed."
}

# Setup Power Systems directories
setup_directories() {
    log "Setting up Power Systems deployment directories..."
    
    mkdir -p "$DEPLOYMENT_DIR/data"
    mkdir -p "$DEPLOYMENT_DIR/config"
    mkdir -p "$DEPLOYMENT_DIR/logs"
    mkdir -p "$DEPLOYMENT_DIR/models"
    mkdir -p "$DEPLOYMENT_DIR/monitoring"
    mkdir -p "$DEPLOYMENT_DIR/compliance"
    
    # Copy models to deployment directory (prefer power systems models)
    if [ -f "$PROJECT_DIR/outputs/models/power_autoencoder.pt" ]; then
        cp "$PROJECT_DIR/outputs/models/power_autoencoder.pt" "$DEPLOYMENT_DIR/models/"
    else
        cp "$PROJECT_DIR/outputs/models/autoencoder.pt" "$DEPLOYMENT_DIR/models/power_autoencoder.pt" 2>/dev/null || true
    fi
    
    if [ -f "$PROJECT_DIR/outputs/models/power_classifier.pt" ]; then
        cp "$PROJECT_DIR/outputs/models/power_classifier.pt" "$DEPLOYMENT_DIR/models/"
    else
        cp "$PROJECT_DIR/outputs/models/classifier.pt" "$DEPLOYMENT_DIR/models/power_classifier.pt" 2>/dev/null || true
    fi
    
    # Copy other models
    cp "$PROJECT_DIR/outputs/models/"*.pt "$DEPLOYMENT_DIR/models/" 2>/dev/null || true
    
    # Create Power Systems configuration files
    cat > "$DEPLOYMENT_DIR/config/power_systems_config.yaml" << EOF
# Power Systems IDS Configuration
substation_id: "$SUBSTATION_ID"
power_systems_mode: $POWER_SYSTEMS_MODE
nerc_cip_mode: $NERC_CIP_MODE

data_sources:
  - name: "scada_events"
    type: "iec61850"
    path: "/app/data/scada_events.jsonl"
    protocol: "IEC 61850 MMS"
    enabled: true
  - name: "protection_relays"
    type: "sel_comtrade"
    path: "/app/data/relay_events.jsonl"
    protocol: "SEL COMTRADE"
    enabled: true
  - name: "pmu_data"
    type: "ieee_c37.118"
    path: "/app/data/pmu_events.jsonl"
    protocol: "IEEE C37.118"
    enabled: true

processing:
  window_size_seconds: 90
  stride_seconds: 20
  resample_step: 1
  input_dim: 196
  power_systems_mode: true
  electrical_features: true
  protection_relay_features: true
  pmu_features: true

autoencoder:
  model_path: "/app/models/power_autoencoder.pt"
  threshold_percentile: 75
  batch_size: 32
  grid_aware_training: true
  substation_id: "$SUBSTATION_ID"

classifier:
  model_path: "/app/models/power_classifier.pt"
  threshold: 0.5
  batch_size: 32
  power_systems_threat_types:
    - "relay_trip_attack"
    - "scada_command_injection"
    - "pmu_data_manipulation"
    - "voltage_stability_attack"
    - "frequency_interference"
    - "protection_zone_bypass"
    - "bay_level_anomaly"
    - "circuit_breaker_misoperation"

alerts:
  enabled: true
  suppression_window: 300  # 5 minutes
  nerc_cip_compliance: true
  channels: ["email", "slack", "scada_hmi", "control_center"]
  grid_escalation:
    critical: ["control_center", "system_operator"]
    high: ["substation_operator", "shift_supervisor"]
    medium: ["maintenance_team", "engineering"]

scada_integration:
  protocols: ["iec61850", "dnp3", "modbus"]
  secure_communication: true
  nerc_cip_compliance: true

pmu_processing:
  ieee_c37_118_enabled: true
  synchrophasor_processing: true
  gps_time_sync: true
  oscillation_detection: true
  frequency_analysis: true

monitoring:
  enabled: true
  grid_dashboard: true
  nerc_cip_dashboard: true
  pmu_integration: true
  metrics_port: 9090
  dashboard_port: 3000

compliance:
  nerc_cip_mode: $NERC_CIP_MODE
  compliance_reporting: true
  audit_logging: true
  security_assessment: true
EOF

    # Create Power Systems Prometheus configuration
    cat > "$DEPLOYMENT_DIR/monitoring/power-prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "power_systems_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'power-autoencoder'
    static_configs:
      - targets: ['power-autoencoder:8083']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'power-classifier'
    static_configs:
      - targets: ['power-classifier:8084']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'power-alert-manager'
    static_configs:
      - targets: ['power-alert-manager:8085']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'scada-integration'
    static_configs:
      - targets: ['scada-integration:24000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'pmu-processor'
    static_configs:
      - targets: ['pmu-processor:24001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'nerc-cip-monitor'
    static_configs:
      - targets: ['nerc-cip-monitor:8086']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF

    # Create Power Systems Grafana dashboard configuration
    mkdir -p "$DEPLOYMENT_DIR/monitoring/grafana/power-dashboards"
    mkdir -p "$DEPLOYMENT_DIR/monitoring/grafana/power-datasources"
    
    cat > "$DEPLOYMENT_DIR/monitoring/grafana/power-datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    log "Power Systems directories setup completed."
}

# Build Power Systems Docker images
build_images() {
    log "Building Power Systems Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build Power Systems specific services
    power_services=("power-data-collector" "power-feature-extractor" "power-autoencoder" "power-classifier" "power-alert-manager" "power-dashboard" "scada-integration" "pmu-processor" "nerc-cip-monitor")
    
    for service in "${power_services[@]}"; do
        log "Building $service service..."
        if ! docker-compose -f "$COMPOSE_FILE" build "$service"; then
            error "Failed to build $service service."
        fi
    done
    
    log "All Power Systems Docker images built successfully."
}

# Deploy Power Systems services
deploy_services() {
    log "Deploying Power Systems IDS services..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Set Power Systems environment variables
    export POWER_SYSTEMS_MODE=$POWER_SYSTEMS_MODE
    export NERC_CIP_MODE=$NERC_CIP_MODE
    export SUBSTATION_ID=$SUBSTATION_ID
    
    # Start services
    if ! docker-compose -f "$COMPOSE_FILE" up -d; then
        error "Failed to start Power Systems services."
    fi
    
    log "Power Systems services deployed successfully."
}

# Wait for Power Systems services to be healthy
wait_for_services() {
    log "Waiting for Power Systems services to be healthy..."
    
    services=(
        "power-autoencoder:8083"
        "power-classifier:8084"
        "power-alert-manager:8085"
        "power-dashboard:8080"
        "scada-integration:24000"
        "pmu-processor:24001"
        "nerc-cip-monitor:8086"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo "$service" | cut -d':' -f1)
        port=$(echo "$service" | cut -d':' -f2)
        
        log "Waiting for $service_name to be healthy..."
        
        max_attempts=30
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if curl -f "http://localhost:$port/health" &> /dev/null; then
                log "$service_name is healthy."
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                warn "$service_name did not become healthy within expected time."
            fi
            
            sleep 2
            ((attempt++))
        done
    done
    
    log "Power Systems service health checks completed."
}

# Run Power Systems deployment tests
run_tests() {
    log "Running Power Systems deployment tests..."
    
    # Test Power Systems autoencoder service
    log "Testing Power Systems autoencoder service..."
    test_features=$(python3 -c "
import numpy as np
print(list(np.random.rand(196).astype(float)))
")
    
    if curl -X POST "http://localhost:8083/predict" \
         -H "Content-Type: application/json" \
         -d "{\"features\": $test_features, \"metadata\": {\"substation_id\": \"$SUBSTATION_ID\", \"voltage_level\": \"230kV\", \"device_type\": \"protection_relay\"}}" &> /dev/null; then
        log "Power Systems autoencoder service test passed."
    else
        warn "Power Systems autoencoder service test failed."
    fi
    
    # Test Power Systems classifier service
    log "Testing Power Systems classifier service..."
    if curl -X POST "http://localhost:8084/classify" \
         -H "Content-Type: application/json" \
         -d "{\"features\": $test_features, \"grid_context\": {\"location\": \"control_center\", \"system_state\": \"normal_operation\"}}" &> /dev/null; then
        log "Power Systems classifier service test passed."
    else
        warn "Power Systems classifier service test failed."
    fi
    
    # Test Power Systems grid metrics
    log "Testing Power Systems grid metrics..."
    if curl -f "http://localhost:8083/metrics/grid" &> /dev/null; then
        log "Power Systems grid metrics test passed."
    else
        warn "Power Systems grid metrics test failed."
    fi
    
    # Test NERC CIP compliance
    if [ "$NERC_CIP_MODE" = "true" ]; then
        log "Testing NERC CIP compliance monitoring..."
        if curl -f "http://localhost:8086/health" &> /dev/null; then
            log "NERC CIP compliance monitor test passed."
        else
            warn "NERC CIP compliance monitor test failed."
        fi
    fi
    
    log "Power Systems deployment tests completed."
}

# Show Power Systems deployment summary
show_summary() {
    log "Power Systems IDS deployment completed successfully!"
    echo
    echo "=== Power Systems IDS Deployment Summary ==="
    echo "Substation ID: $SUBSTATION_ID"
    echo "Power Systems Mode: $POWER_SYSTEMS_MODE"
    echo "NERC CIP Mode: $NERC_CIP_MODE"
    echo
    echo "Access Points:"
    echo "  Power Dashboard: http://localhost:8080"
    echo "  Grid Monitoring: http://localhost:8080/grid"
    echo "  Grafana Power Metrics: http://localhost:3000 (grid-admin123)"
    echo "  Prometheus: http://localhost:9090"
    echo "  NERC CIP Compliance: http://localhost:8086"
    echo
    echo "Power Systems Service Endpoints:"
    echo "  Power Autoencoder: http://localhost:8083"
    echo "  Power Classifier: http://localhost:8084"
    echo "  Power Alert Manager: http://localhost:8085"
    echo "  SCADA Integration: http://localhost:24000"
    echo "  PMU Processor: http://localhost:24001"
    echo
    echo "Management Commands:"
    echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f [service]"
    echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo
    echo "Power Systems Monitoring:"
    echo "  Check service health: curl http://localhost:8083/health"
    echo "  View grid metrics: curl http://localhost:8083/metrics/grid"
    echo "  Check NERC CIP compliance: curl http://localhost:8086/compliance/nerc"
    echo
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        error "Power Systems deployment failed. Check logs for details."
    fi
}

# Main Power Systems deployment function
main() {
    log "Starting Power Systems Industrial IDS deployment..."
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    setup_directories
    build_images
    deploy_services
    wait_for_services
    run_tests
    show_summary
    
    log "Power Systems IDS deployment completed successfully!"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping Power Systems IDS services..."
        cd "$DEPLOYMENT_DIR"
        docker-compose down
        log "Power Systems services stopped."
        ;;
    "restart")
        log "Restarting Power Systems IDS services..."
        cd "$DEPLOYMENT_DIR"
        docker-compose restart
        log "Power Systems services restarted."
        ;;
    "logs")
        cd "$DEPLOYMENT_DIR"
        docker-compose logs -f "${2:-}"
        ;;
    "status")
        cd "$DEPLOYMENT_DIR"
        docker-compose ps
        ;;
    "test")
        run_tests
        ;;
    "compliance")
        if [ "$NERC_CIP_MODE" = "true" ]; then
            log "Checking NERC CIP compliance status..."
            curl -s "http://localhost:8086/compliance/nerc" | python3 -m json.tool
        else
            warn "NERC CIP mode is disabled. Enable with NERC_CIP_MODE=true"
        fi
        ;;
    "grid-status")
        log "Checking Power Systems grid status..."
        curl -s "http://localhost:8083/metrics/grid" | python3 -m json.tool
        ;;
    "cleanup")
        log "Cleaning up Power Systems deployment..."
        cd "$DEPLOYMENT_DIR"
        docker-compose down -v
        docker system prune -f
        log "Power Systems cleanup completed."
        ;;
    "help"|"-h"|"--help")
        echo "Power Systems Industrial IDS Deployment Script"
        echo
        echo "Usage: $0 [COMMAND]"
        echo
        echo "Commands:"
        echo "  deploy        Deploy the Power Systems IDS (default)"
        echo "  stop          Stop all Power Systems services"
        echo "  restart       Restart all Power Systems services"
        echo "  logs          Show logs for all or specific service"
        echo "  status        Show Power Systems service status"
        echo "  test          Run Power Systems deployment tests"
        echo "  compliance    Check NERC CIP compliance status"
        echo "  grid-status   Check Power Systems grid status"
        echo "  cleanup       Remove all containers and data"
        echo "  help          Show this help message"
        echo
        echo "Environment Variables:"
        echo "  POWER_SYSTEMS_MODE    Enable Power Systems features (default: true)"
        echo "  NERC_CIP_MODE         Enable NERC CIP compliance (default: true)"
        echo "  SUBSTATION_ID         Substation identifier (default: SUB001)"
        echo
        echo "Examples:"
        echo "  $0 deploy                    # Deploy Power Systems IDS"
        echo "  $0 logs power-autoencoder     # Show autoencoder logs"
        echo "  $0 grid-status                # Check grid status"
        echo "  $0 compliance                 # Check NERC CIP compliance"
        echo "  SUBSTATION_ID=SUB002 $0 deploy # Deploy for substation SUB002"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        ;;
esac
