#!/bin/bash

# AI Data Readiness Platform Rollback Script
set -e

NAMESPACE="ai-data-readiness"
DEPLOYMENT_NAME="ai-data-readiness-app"

echo "🔄 Starting rollback process..."

# Function to get rollout history
get_rollout_history() {
    kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE
}

# Function to rollback to previous version
rollback_to_previous() {
    echo "🔄 Rolling back to previous version..."
    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE
    
    echo "⏳ Waiting for rollback to complete..."
    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s
    
    echo "✅ Rollback completed successfully!"
}

# Function to rollback to specific revision
rollback_to_revision() {
    local revision=$1
    echo "🔄 Rolling back to revision $revision..."
    kubectl rollout undo deployment/$DEPLOYMENT_NAME --to-revision=$revision -n $NAMESPACE
    
    echo "⏳ Waiting for rollback to complete..."
    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s
    
    echo "✅ Rollback to revision $revision completed successfully!"
}

# Check if revision is specified
if [ $# -eq 0 ]; then
    echo "📋 Current rollout history:"
    get_rollout_history
    echo ""
    echo "🔄 Rolling back to previous version..."
    rollback_to_previous
elif [ $# -eq 1 ]; then
    revision=$1
    echo "📋 Current rollout history:"
    get_rollout_history
    echo ""
    rollback_to_revision $revision
else
    echo "Usage: $0 [revision_number]"
    echo "  No arguments: rollback to previous version"
    echo "  revision_number: rollback to specific revision"
    exit 1
fi

# Verify rollback
echo "📊 Current deployment status:"
kubectl get pods -n $NAMESPACE -l app=ai-data-readiness
kubectl describe deployment/$DEPLOYMENT_NAME -n $NAMESPACE | grep Image: