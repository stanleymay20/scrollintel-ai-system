"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  RefreshCw, 
  X,
  Info,
  Clock
} from 'lucide-react';

interface ValidationError {
  type: 'error' | 'warning' | 'info';
  message: string;
  nodeId?: string;
  connectionId?: string;
}

interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  timestamp: string;
  executionTime: number;
}

interface PipelineValidationProps {
  pipelineId?: string;
  onClose: () => void;
}

export function PipelineValidation({ pipelineId, onClose }: PipelineValidationProps) {
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [autoValidate, setAutoValidate] = useState(true);

  const validatePipeline = async () => {
    if (!pipelineId) return;

    setIsValidating(true);
    const startTime = Date.now();

    try {
      const response = await fetch(`/api/pipelines/${pipelineId}/validate`, {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        const executionTime = Date.now() - startTime;
        
        setValidationResult({
          isValid: result.is_valid,
          errors: result.errors?.map((error: string) => ({
            type: 'error' as const,
            message: error,
          })) || [],
          warnings: result.warnings?.map((warning: string) => ({
            type: 'warning' as const,
            message: warning,
          })) || [],
          timestamp: new Date().toISOString(),
          executionTime,
        });
      }
    } catch (error) {
      console.error('Validation failed:', error);
      setValidationResult({
        isValid: false,
        errors: [{
          type: 'error',
          message: 'Failed to validate pipeline. Please try again.',
        }],
        warnings: [],
        timestamp: new Date().toISOString(),
        executionTime: Date.now() - startTime,
      });
    } finally {
      setIsValidating(false);
    }
  };

  useEffect(() => {
    if (pipelineId && autoValidate) {
      validatePipeline();
    }
  }, [pipelineId, autoValidate]);

  const getStatusIcon = () => {
    if (isValidating) {
      return <RefreshCw className="w-5 h-5 animate-spin text-blue-500" />;
    }
    
    if (!validationResult) {
      return <Clock className="w-5 h-5 text-gray-400" />;
    }

    if (validationResult.isValid) {
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    }

    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const getStatusText = () => {
    if (isValidating) return 'Validating...';
    if (!validationResult) return 'Not validated';
    if (validationResult.isValid) return 'Valid';
    return 'Invalid';
  };

  const getStatusColor = () => {
    if (isValidating) return 'text-blue-600';
    if (!validationResult) return 'text-gray-500';
    if (validationResult.isValid) return 'text-green-600';
    return 'text-red-600';
  };

  const allIssues = [
    ...(validationResult?.errors || []),
    ...(validationResult?.warnings || []),
  ];

  const getIssueIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      default:
        return <Info className="w-4 h-4 text-blue-500" />;
    }
  };

  const getIssueColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'border-red-200 bg-red-50';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50';
      default:
        return 'border-blue-200 bg-blue-50';
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <div>
              <h3 className="font-semibold">Pipeline Validation</h3>
              <p className={`text-sm ${getStatusColor()}`}>
                {getStatusText()}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Validation Summary */}
      {validationResult && (
        <div className="p-4 border-b bg-gray-50">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Status:</span>
              <Badge 
                variant={validationResult.isValid ? 'default' : 'destructive'}
                className="ml-2"
              >
                {validationResult.isValid ? 'Valid' : 'Invalid'}
              </Badge>
            </div>
            <div>
              <span className="text-gray-600">Execution Time:</span>
              <span className="ml-2 font-mono">
                {validationResult.executionTime}ms
              </span>
            </div>
            <div>
              <span className="text-gray-600">Errors:</span>
              <span className="ml-2 font-semibold text-red-600">
                {validationResult.errors.length}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Warnings:</span>
              <span className="ml-2 font-semibold text-yellow-600">
                {validationResult.warnings.length}
              </span>
            </div>
          </div>
          
          {validationResult.timestamp && (
            <p className="text-xs text-gray-500 mt-2">
              Last validated: {new Date(validationResult.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      )}

      {/* Issues List */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {allIssues.length === 0 && validationResult && (
            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <div>
                    <p className="font-medium text-green-800">
                      Pipeline is valid!
                    </p>
                    <p className="text-sm text-green-600">
                      No issues found. Your pipeline is ready to run.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {allIssues.map((issue, index) => (
            <Card key={index} className={`border ${getIssueColor(issue.type)}`}>
              <CardContent className="p-3">
                <div className="flex items-start space-x-2">
                  {getIssueIcon(issue.type)}
                  <div className="flex-1">
                    <p className="text-sm font-medium capitalize">
                      {issue.type}
                    </p>
                    <p className="text-sm text-gray-700 mt-1">
                      {issue.message}
                    </p>
                    {(issue.nodeId || issue.connectionId) && (
                      <p className="text-xs text-gray-500 mt-1">
                        {issue.nodeId && `Node: ${issue.nodeId}`}
                        {issue.connectionId && `Connection: ${issue.connectionId}`}
                      </p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}

          {!validationResult && !isValidating && (
            <div className="text-center py-8 text-gray-500">
              <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No validation results</p>
              <p className="text-xs">Click validate to check your pipeline</p>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Actions */}
      <div className="p-4 border-t bg-white space-y-2">
        <Button 
          onClick={validatePipeline} 
          disabled={isValidating || !pipelineId}
          className="w-full"
        >
          {isValidating ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Validating...
            </>
          ) : (
            <>
              <RefreshCw className="w-4 h-4 mr-2" />
              Validate Pipeline
            </>
          )}
        </Button>
        
        <div className="flex items-center justify-between text-sm">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoValidate}
              onChange={(e) => setAutoValidate(e.target.checked)}
              className="rounded"
            />
            <span className="text-gray-600">Auto-validate</span>
          </label>
        </div>
      </div>
    </div>
  );
}