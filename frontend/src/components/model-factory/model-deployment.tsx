"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, CheckCircle, AlertCircle, Rocket, Copy, ExternalLink, Code } from 'lucide-react';
import { api } from '@/lib/api';

interface ModelDeploymentProps {
  modelId: string;
  modelName: string;
  onDeploymentComplete?: (result: any) => void;
}

export function ModelDeployment({ modelId, modelName, onDeploymentComplete }: ModelDeploymentProps) {
  const [deploying, setDeploying] = useState(false);
  const [endpointName, setEndpointName] = useState('');
  const [deploymentResult, setDeploymentResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [testingPrediction, setTestingPrediction] = useState(false);
  const [testFeatures, setTestFeatures] = useState('');
  const [predictionResult, setPredictionResult] = useState<any>(null);

  const handleDeploy = async () => {
    try {
      setDeploying(true);
      setError(null);

      const requestData = {
        model_id: modelId,
        endpoint_name: endpointName || undefined
      };

      const response = await api.post(`/model-factory/models/${modelId}/deploy`, requestData);
      
      setDeploymentResult(response.data);
      onDeploymentComplete?.(response.data);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Deployment failed');
    } finally {
      setDeploying(false);
    }
  };

  const handleTestPrediction = async () => {
    if (!deploymentResult) return;

    try {
      setTestingPrediction(true);
      setPredictionResult(null);

      let features;
      try {
        features = JSON.parse(testFeatures);
        if (!Array.isArray(features)) {
          throw new Error('Features must be an array');
        }
      } catch (parseError) {
        throw new Error('Invalid JSON format for features');
      }

      const response = await api.get(`/model-factory/models/${modelId}/predict`, {
        params: { features: features.join(',') }
      });
      
      setPredictionResult(response.data);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction test failed');
    } finally {
      setTestingPrediction(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const generateCurlExample = () => {
    if (!deploymentResult) return '';
    
    return `curl -X GET "${window.location.origin}${deploymentResult.api_endpoint}?features=1.0,2.0,3.0" \\
  -H "Authorization: Bearer YOUR_TOKEN"`;
  };

  const generatePythonExample = () => {
    if (!deploymentResult) return '';
    
    return `import requests

url = "${window.location.origin}${deploymentResult.api_endpoint}"
headers = {"Authorization": "Bearer YOUR_TOKEN"}
params = {"features": "1.0,2.0,3.0"}

response = requests.get(url, headers=headers, params=params)
result = response.json()
print(result)`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Model Deployment</h3>
          <p className="text-sm text-muted-foreground">
            Deploy model: {modelName}
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

      {!deploymentResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5" />
              Deploy Model
            </CardTitle>
            <CardDescription>
              Deploy your model to create an API endpoint for predictions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="endpoint-name">Endpoint Name (optional)</Label>
              <Input
                id="endpoint-name"
                value={endpointName}
                onChange={(e) => setEndpointName(e.target.value)}
                placeholder="my-custom-model"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to use default naming
              </p>
            </div>

            <Button 
              onClick={handleDeploy} 
              disabled={deploying}
              className="w-full"
            >
              {deploying ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deploying...
                </>
              ) : (
                <>
                  <Rocket className="mr-2 h-4 w-4" />
                  Deploy Model
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {deploymentResult && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Deployment Successful
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="font-medium">Endpoint Name</Label>
                  <p className="text-sm">{deploymentResult.endpoint_name}</p>
                </div>
                
                <div className="space-y-2">
                  <Label className="font-medium">Status</Label>
                  <Badge variant="default">{deploymentResult.status}</Badge>
                </div>
                
                <div className="space-y-2">
                  <Label className="font-medium">Deployment Time</Label>
                  <p className="text-sm text-muted-foreground">
                    {new Date(deploymentResult.deployment_timestamp).toLocaleString()}
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="font-medium">API Endpoint</Label>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-2 rounded text-sm">
                    {deploymentResult.api_endpoint}
                  </code>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => copyToClipboard(deploymentResult.api_endpoint)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ExternalLink className="h-5 w-5" />
                Test Prediction
              </CardTitle>
              <CardDescription>
                Test your deployed model with sample data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="test-features">Test Features (JSON Array)</Label>
                <Input
                  id="test-features"
                  value={testFeatures}
                  onChange={(e) => setTestFeatures(e.target.value)}
                  placeholder="[1.0, 2.0, 3.0, 4.0]"
                />
                <p className="text-xs text-muted-foreground">
                  Provide feature values as a JSON array
                </p>
              </div>

              <Button 
                onClick={handleTestPrediction} 
                disabled={testingPrediction || !testFeatures}
                className="w-full"
              >
                {testingPrediction ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Testing...
                  </>
                ) : (
                  <>
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Test Prediction
                  </>
                )}
              </Button>

              {predictionResult && (
                <div className="space-y-2">
                  <Label className="font-medium">Prediction Result</Label>
                  <div className="bg-muted p-3 rounded-md">
                    <pre className="text-sm overflow-x-auto">
                      {JSON.stringify(predictionResult, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                API Usage Examples
              </CardTitle>
              <CardDescription>
                Code examples for using your deployed model
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="font-medium">cURL</Label>
                <div className="relative">
                  <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                    {generateCurlExample()}
                  </pre>
                  <Button
                    size="sm"
                    variant="outline"
                    className="absolute top-2 right-2"
                    onClick={() => copyToClipboard(generateCurlExample())}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="font-medium">Python</Label>
                <div className="relative">
                  <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                    {generatePythonExample()}
                  </pre>
                  <Button
                    size="sm"
                    variant="outline"
                    className="absolute top-2 right-2"
                    onClick={() => copyToClipboard(generatePythonExample())}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Replace YOUR_TOKEN with your actual API token for authentication.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}