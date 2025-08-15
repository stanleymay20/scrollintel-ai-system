"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, CheckCircle, AlertCircle, Play, FileText } from 'lucide-react';
import { api } from '@/lib/api';

interface ModelValidationProps {
  modelId: string;
  modelName: string;
  onValidationComplete?: (result: any) => void;
}

export function ModelValidation({ modelId, modelName, onValidationComplete }: ModelValidationProps) {
  const [validating, setValidating] = useState(false);
  const [validationData, setValidationData] = useState('');
  const [validationResult, setValidationResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleValidate = async () => {
    try {
      setValidating(true);
      setError(null);

      let parsedData = null;
      if (validationData.trim()) {
        try {
          // Try to parse as JSON array
          parsedData = JSON.parse(validationData);
          if (!Array.isArray(parsedData)) {
            throw new Error('Validation data must be an array');
          }
        } catch (parseError) {
          throw new Error('Invalid JSON format. Please provide data as a JSON array.');
        }
      }

      const requestData = {
        model_id: modelId,
        validation_data: parsedData
      };

      const response = await api.post(`/model-factory/models/${modelId}/validate`, requestData);
      
      setValidationResult(response.data);
      onValidationComplete?.(response.data);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Validation failed');
    } finally {
      setValidating(false);
    }
  };

  const handleClearResults = () => {
    setValidationResult(null);
    setError(null);
    setValidationData('');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Model Validation</h3>
          <p className="text-sm text-muted-foreground">
            Validate model: {modelName}
          </p>
        </div>
        <Badge variant="outline">{modelId}</Badge>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Validation Data
          </CardTitle>
          <CardDescription>
            Provide validation data as a JSON array (optional). Leave empty to just validate model loading.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="validation-data">Validation Data (JSON Array)</Label>
            <Textarea
              id="validation-data"
              value={validationData}
              onChange={(e) => setValidationData(e.target.value)}
              placeholder='[
  [1.2, 3.4, 5.6, 7.8],
  [2.1, 4.3, 6.5, 8.7],
  [3.0, 5.2, 7.4, 9.6]
]'
              rows={8}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Each inner array should contain feature values in the same order as training data
            </p>
          </div>

          <div className="flex gap-2">
            <Button 
              onClick={handleValidate} 
              disabled={validating}
              className="flex-1"
            >
              {validating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Validating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Validate Model
                </>
              )}
            </Button>
            
            {(validationResult || error) && (
              <Button onClick={handleClearResults} variant="outline">
                Clear Results
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {validationResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Validation Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="font-medium">Status</Label>
                <Badge 
                  variant={validationResult.validation_status === 'success' ? 'default' : 'secondary'}
                >
                  {validationResult.validation_status}
                </Badge>
              </div>
              
              <div className="space-y-2">
                <Label className="font-medium">Timestamp</Label>
                <p className="text-sm text-muted-foreground">
                  {new Date(validationResult.validation_timestamp).toLocaleString()}
                </p>
              </div>
            </div>

            {validationResult.predictions && (
              <div className="space-y-2">
                <Label className="font-medium">Predictions</Label>
                <div className="bg-muted p-3 rounded-md">
                  <pre className="text-sm overflow-x-auto">
                    {JSON.stringify(validationResult.predictions, null, 2)}
                  </pre>
                </div>
                <p className="text-xs text-muted-foreground">
                  {validationResult.predictions.length} predictions generated
                </p>
              </div>
            )}

            {validationResult.validation_status === 'model_loaded' && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Model loaded successfully and is ready for predictions.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Validation Tips</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <ul className="text-sm space-y-1 text-muted-foreground">
            <li>• Leave validation data empty to just test if the model loads correctly</li>
            <li>• Provide validation data as a JSON array of arrays</li>
            <li>• Each inner array should contain feature values in the same order as training</li>
            <li>• The number of features should match the training data</li>
            <li>• Use this to test model performance on new, unseen data</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}