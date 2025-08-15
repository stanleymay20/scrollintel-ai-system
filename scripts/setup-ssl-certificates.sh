#!/bin/bash

# ================================
# SSL Certificate Setup Script
# ================================

set -e

# Configuration
DOMAIN="scrollintel.com"
EMAIL="admin@scrollintel.com"
SSL_DIR="/etc/nginx/ssl"
CERTBOT_DIR="/etc/letsencrypt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
fi

# Create SSL directory
log "Creating SSL directory..."
mkdir -p $SSL_DIR
chmod 700 $SSL_DIR

# Install certbot if not present
if ! command -v certbot &> /dev/null; then
    log "Installing certbot..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y certbot python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        yum install -y certbot python3-certbot-nginx
    else
        error "Package manager not supported. Please install certbot manually."
    fi
fi

# Generate DH parameters for perfect forward secrecy
if [ ! -f "$SSL_DIR/dhparam.pem" ]; then
    log "Generating DH parameters (this may take a while)..."
    openssl dhparam -out $SSL_DIR/dhparam.pem 2048
    chmod 600 $SSL_DIR/dhparam.pem
else
    log "DH parameters already exist"
fi

# Function to obtain SSL certificate
obtain_certificate() {
    local domain=$1
    log "Obtaining SSL certificate for $domain..."
    
    # Stop nginx temporarily
    systemctl stop nginx || true
    
    # Obtain certificate
    certbot certonly \
        --standalone \
        --email $EMAIL \
        --agree-tos \
        --no-eff-email \
        --domains $domain,www.$domain \
        --non-interactive
    
    # Create symlinks in nginx SSL directory
    ln -sf $CERTBOT_DIR/live/$domain/fullchain.pem $SSL_DIR/scrollintel.crt
    ln -sf $CERTBOT_DIR/live/$domain/privkey.pem $SSL_DIR/scrollintel.key
    
    # Set proper permissions
    chmod 644 $SSL_DIR/scrollintel.crt
    chmod 600 $SSL_DIR/scrollintel.key
    
    # Start nginx
    systemctl start nginx
    
    log "SSL certificate obtained successfully for $domain"
}

# Function to create self-signed certificate for development
create_self_signed() {
    log "Creating self-signed certificate for development..."
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout $SSL_DIR/scrollintel.key \
        -out $SSL_DIR/scrollintel.crt \
        -subj "/C=US/ST=CA/L=San Francisco/O=ScrollIntel/CN=scrollintel.local"
    
    chmod 644 $SSL_DIR/scrollintel.crt
    chmod 600 $SSL_DIR/scrollintel.key
    
    log "Self-signed certificate created"
}

# Function to setup certificate auto-renewal
setup_auto_renewal() {
    log "Setting up certificate auto-renewal..."
    
    # Create renewal script
    cat > /usr/local/bin/renew-ssl.sh << 'EOF'
#!/bin/bash
certbot renew --quiet --nginx
systemctl reload nginx
EOF
    
    chmod +x /usr/local/bin/renew-ssl.sh
    
    # Add cron job for auto-renewal
    (crontab -l 2>/dev/null; echo "0 3 * * * /usr/local/bin/renew-ssl.sh") | crontab -
    
    log "Auto-renewal configured"
}

# Main execution
case "${1:-production}" in
    "production")
        log "Setting up production SSL certificates..."
        obtain_certificate $DOMAIN
        setup_auto_renewal
        ;;
    "development")
        log "Setting up development SSL certificates..."
        create_self_signed
        ;;
    "renew")
        log "Renewing SSL certificates..."
        certbot renew --nginx
        systemctl reload nginx
        ;;
    *)
        echo "Usage: $0 [production|development|renew]"
        exit 1
        ;;
esac

log "SSL certificate setup completed successfully!"