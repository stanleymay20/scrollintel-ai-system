#!/bin/bash

# ================================
# ScrollIntel™ Launch Script (Unix/Linux/Mac)
# One-click setup and launch for ScrollIntel AI Platform
# ================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${PURPLE}"
cat << "EOF"
   _____ _____ _____   ____  _      _      _____ _   _ _______ ______ _      
  / ____/ ____|  __ \ / __ \| |    | |    |_   _| \ | |__   __|  ____| |     
 | (___| |    | |__) | |  | | |    | |      | | |  \| |  | |  | |__  | |     
  \___ \ |    |  _  /| |  | | |    | |      | | | . ` |  | |  |  __| | |     
  ____) | |____| | \ \| |__| | |____| |____ _| |_| |\  |  | |  | |____| |____ 
 |_____/ \_____|_|  \_\\____/|______|______|_____|_| \_|  |_|  |______|______|
                                                                              
EOF
echo -e "${NC}"

echo -e "${CYAN}🚀 ScrollIntel™ AI Platform Launcher${NC}"
echo -e "${BLUE}Replace your CTO with AI agents that analyze data, build models, and make technical decisions${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first:"
        echo "  - Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first:"
        echo "  - Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are ready!"
}

# Generate secure JWT secret
generate_jwt_secret() {
    if command -v openssl &> /dev/null; then
        openssl rand -hex 32
    elif command -v python3 &> /dev/null; then
        python3 -c "import secrets; print(secrets.token_hex(32))"
    else
        # Fallback to a reasonably secure random string
        date +%s | sha256sum | base64 | head -c 64
    fi
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            print_error ".env.example file not found!"
            exit 1
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Generate JWT secret if not set
    if ! grep -q "JWT_SECRET_KEY=.*[a-zA-Z0-9]" .env; then
        JWT_SECRET=$(generate_jwt_secret)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
        else
            # Linux
            sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
        fi
        print_success "Generated secure JWT secret"
    fi
    
    # Set default database password if not set
    if ! grep -q "POSTGRES_PASSWORD=.*[a-zA-Z0-9]" .env; then
        DB_PASSWORD=$(generate_jwt_secret | head -c 16)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$DB_PASSWORD/" .env
        else
            # Linux
            sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$DB_PASSWORD/" .env
        fi
        print_success "Generated database password"
    fi
}

# Check for required API keys
check_api_keys() {
    print_status "Checking API key configuration..."
    
    if ! grep -q "OPENAI_API_KEY=sk-" .env; then
        print_warning "OpenAI API key not configured in .env file"
        echo -e "${YELLOW}  To enable AI features, add your OpenAI API key to .env:${NC}"
        echo -e "${YELLOW}  OPENAI_API_KEY=sk-your-key-here${NC}"
        echo ""
    else
        print_success "OpenAI API key configured"
    fi
}

# Start services
start_services() {
    print_status "Starting ScrollIntel services..."
    
    # Pull latest images
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Build and start services
    print_status "Building and starting containers..."
    docker-compose up -d --build
    
    print_success "All services started!"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T postgres pg_isready -U postgres &> /dev/null; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "PostgreSQL failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for backend API
    print_status "Waiting for backend API..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Backend API not responding, but continuing..."
    else
        print_success "Backend API is ready!"
    fi
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:3000 &> /dev/null; then
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_warning "Frontend not responding, but continuing..."
    else
        print_success "Frontend is ready!"
    fi
}

# Display success information
show_success() {
    echo ""
    echo -e "${GREEN}🎉 ScrollIntel™ is now running!${NC}"
    echo ""
    echo -e "${CYAN}📱 Access Points:${NC}"
    echo -e "  🌐 Frontend:    ${BLUE}http://localhost:3000${NC}"
    echo -e "  🔧 API:         ${BLUE}http://localhost:8000${NC}"
    echo -e "  📚 API Docs:    ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  ❤️  Health:     ${BLUE}http://localhost:8000/health${NC}"
    echo ""
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    echo -e "  1. Open ${BLUE}http://localhost:3000${NC} in your browser"
    echo -e "  2. Upload your data files (CSV, Excel, JSON)"
    echo -e "  3. Chat with AI agents for insights"
    echo -e "  4. Build ML models with AutoML"
    echo -e "  5. Create interactive dashboards"
    echo ""
    echo -e "${CYAN}🛠️  Management:${NC}"
    echo -e "  📊 View logs:   ${YELLOW}docker-compose logs -f${NC}"
    echo -e "  🔄 Restart:     ${YELLOW}docker-compose restart${NC}"
    echo -e "  🛑 Stop:        ${YELLOW}docker-compose down${NC}"
    echo ""
    echo -e "${PURPLE}ScrollIntel™ - Where artificial intelligence meets unlimited potential! 🌟${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting ScrollIntel™ launch sequence...${NC}"
    echo ""
    
    check_docker
    setup_environment
    check_api_keys
    start_services
    wait_for_services
    show_success
}

# Run main function
main "$@"