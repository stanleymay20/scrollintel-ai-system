"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  validationId?: string;
}

interface PipelineValidationProps {
  result: ValidationResult;
}

export function PipelineValidation({ result }: PipelineValidationProps) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle className="text-sm flex items-center space-x-2">
          <span>Validation Results</span>
          <Badge variant={result.isValid ? "default" : "destructive"}>
            {result.isValid ? "Valid" : "Invalid"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Validation Status */}
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Status:</span>
          <div className="flex items-center space-x-1">
            <span className={`w-2 h-2 rounded-full ${result.isValid ? 'bg-green-500' : 'bg-red-500'}`}></span>
            <span className="text-sm">
              {result.isValid ? 'Pipeline is valid' : 'Pipeline has issues'}
            </span>
          </div>
        </div>

        {/* Errors */}
        {result.errors && result.errors.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-red-700 mb-2">
              Errors ({result.errors.length})
            </h5>
            <div className="space-y-2">
              {result.errors.map((error, index) => (
                <Alert key={index} variant="destructive">
                  <AlertDescription className="text-xs">
                    {error}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </div>
        )}

        {/* Warnings */}
        {result.warnings && result.warnings.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-yellow-700 mb-2">
              Warnings ({result.warnings.length})
            </h5>
            <div className="space-y-2">
              {result.warnings.map((warning, index) => (
                <Alert key={index} className="border-yellow-200 bg-yellow-50">
                  <AlertDescription className="text-xs text-yellow-800">
                    {warning}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </div>
        )}

        {/* Success Message */}
        {result.isValid && result.errors.length === 0 && result.warnings.length === 0 && (
          <Alert className="border-green-200 bg-green-50">
            <AlertDescription className="text-xs text-green-800">
              ✅ Pipeline validation passed successfully! Your pipeline is ready to run.
            </AlertDescription>
          </Alert>
        )}

        {/* Validation Summary */}
        <div className="pt-2 border-t">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="font-medium">Errors:</span>
              <Badge variant="destructive" className="ml-1">
                {result.errors?.length || 0}
              </Badge>
            </div>
            <div>
              <span className="font-medium">Warnings:</span>
              <Badge variant="secondary" className="ml-1">
                {result.warnings?.length || 0}
              </Badge>
            </div>
          </div>
          
          {result.validationId && (
            <div className="mt-2 text-xs text-gray-500">
              Validation ID: {result.validationId}
            </div>
          )}
        </div>

        {/* Validation Tips */}
        {!result.isValid && (
          <div className="mt-3 p-2 bg-blue-50 rounded">
            <h6 className="text-xs font-medium text-blue-700 mb-1">💡 Tips to fix issues:</h6>
            <ul className="text-xs text-blue-600 space-y-1">
              {result.errors?.some(e => e.includes('data source')) && (
                <li>• Add at least one data source node to your pipeline</li>
              )}
              {result.errors?.some(e => e.includes('circular')) && (
                <li>• Remove circular connections between nodes</li>
              )}
              {result.warnings?.some(w => w.includes('disconnected')) && (
                <li>• Connect all nodes to create a complete data flow</li>
              )}
              {result.warnings?.some(w => w.includes('target')) && (
                <li>• Add a data target to store your processed data</li>
              )}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}