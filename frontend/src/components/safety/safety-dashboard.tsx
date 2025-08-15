"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  AlertTriangle, 
  Users, 
  CheckCircle, 
  XCircle, 
  Clock,
  Activity,
  Zap,
  Eye,
  Settings
} from 'lucide-react';

interface SafetyStatus {
  safety_active: boolean;
  shutdown_active: boolean;
  total_constraints: number;
  active_constraints: number;
  total_violations: number;
  unresolved_violations: number;
  pending_approvals: number;
  human_overseers: number;
  alignment_checks: number;
  last_alignment_check: string | null;
}

interface SafetyViolation {
  id: string;
  constraint_id: string;
  violation_type: string;
  severity: string;
  description: string;
  resolved: boolean;
  human_notified: boolean;
  timestamp: string;
}

interface AlignmentCheck {
  id: string;
  check_type: string;
  description: string;
  status: string;
  confidence_score: number;
  human_verified: boolean;
  timestamp: string;
}

interface HumanOverseer {
  id: string;
  name: string;
  role: string;
  clearance_level: string;
  permissions: string[];
  active: boolean;
  last_active: string;
}

interface MiddlewareStats {
  enabled: boolean;
  total_operations: number;
  completed_operations: number;
  blocked_operations: number;
  error_operations: number;
  success_rate: number;
  block_rate: number;
}

export default function SafetyDashboard() {
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus | null>(null);
  const [violations, setViolations] = useState<SafetyViolation[]>([]);
  const [alignmentChecks, setAlignmentChecks] = useState<AlignmentCheck[]>([]);
  const [overseers, setOverseers] = useState<HumanOverseer[]>([]);
  const [middlewareStats, setMiddlewareStats] = useState<MiddlewareStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSafetyData();
    const interval = setInterval(fetchSafetyData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchSafetyData = async () => {
    try {
      setLoading(true);
      
      // Fetch all safety data in parallel
      const [statusRes, violationsRes, alignmentRes, overseersRes, middlewareRes] = await Promise.all([
        fetch('/api/safety/status'),
        fetch('/api/safety/violations'),
        fetch('/api/safety/alignment-checks'),
        fetch('/api/safety/human-overseers'),
        fetch('/api/safety/middleware-stats')
      ]);

      if (!statusRes.ok) throw new Error('Failed to fetch safety status');
      
      const statusData = await statusRes.json();
      setSafetyStatus(statusData);

      if (violationsRes.ok) {
        const violationsData = await violationsRes.json();
        setViolations(violationsData.data.violations);
      }

      if (alignmentRes.ok) {
        const alignmentData = await alignmentRes.json();
        setAlignmentChecks(alignmentData.data.checks);
      }

      if (overseersRes.ok) {
        const overseersData = await overseersRes.json();
        setOverseers(overseersData.data.overseers);
      }

      if (middlewareRes.ok) {
        const middlewareData = await middlewareRes.json();
        setMiddlewareStats(middlewareData.data);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch safety data');
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyShutdown = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to initiate emergency shutdown? This will stop all AI operations immediately.'
    );
    
    if (!confirmed) return;

    const reason = window.prompt('Please provide a reason for emergency shutdown:');
    if (!reason) return;

    const confirmationCode = window.prompt('Enter confirmation code (EMERGENCY_SHUTDOWN_CONFIRMED):');
    if (confirmationCode !== 'EMERGENCY_SHUTDOWN_CONFIRMED') {
      alert('Invalid confirmation code');
      return;
    }

    try {
      const response = await fetch('/api/safety/emergency-shutdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason,
          authorized_user: 'dashboard_user',
          confirmation_code: confirmationCode
        })
      });

      if (response.ok) {
        alert('Emergency shutdown initiated successfully');
        fetchSafetyData();
      } else {
        const error = await response.json();
        alert(`Emergency shutdown failed: ${error.detail}`);
      }
    } catch (err) {
      alert(`Emergency shutdown failed: ${err}`);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'existential': return 'bg-red-600';
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getAlignmentColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'aligned': return 'bg-green-500';
      case 'misaligned': return 'bg-red-500';
      case 'uncertain': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  if (loading && !safetyStatus) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Activity className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading safety dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading safety dashboard: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Safety Dashboard</h1>
          <p className="text-gray-600">Monitor and control AI safety systems</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="destructive" 
            onClick={handleEmergencyShutdown}
            className="bg-red-600 hover:bg-red-700"
          >
            <Zap className="h-4 w-4 mr-2" />
            Emergency Shutdown
          </Button>
          <Button variant="outline" onClick={fetchSafetyData}>
            <Activity className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Critical Status Alerts */}
      {safetyStatus?.shutdown_active && (
        <Alert className="border-red-500 bg-red-50">
          <XCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            <strong>EMERGENCY SHUTDOWN ACTIVE</strong> - All AI operations have been halted
          </AlertDescription>
        </Alert>
      )}

      {!safetyStatus?.safety_active && (
        <Alert className="border-red-500 bg-red-50">
          <AlertTriangle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            <strong>SAFETY FRAMEWORK DISABLED</strong> - System is operating without safety constraints
          </AlertDescription>
        </Alert>
      )}

      {safetyStatus && safetyStatus.unresolved_violations > 0 && (
        <Alert className="border-orange-500 bg-orange-50">
          <AlertTriangle className="h-4 w-4 text-orange-600" />
          <AlertDescription className="text-orange-800">
            <strong>{safetyStatus.unresolved_violations} unresolved safety violations</strong> require immediate attention
          </AlertDescription>
        </Alert>
      )}

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Safety Status</CardTitle>
            <Shield className={`h-4 w-4 ${safetyStatus?.safety_active ? 'text-green-600' : 'text-red-600'}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {safetyStatus?.safety_active ? 'ACTIVE' : 'DISABLED'}
            </div>
            <p className="text-xs text-gray-600">
              {safetyStatus?.active_constraints}/{safetyStatus?.total_constraints} constraints active
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Safety Violations</CardTitle>
            <AlertTriangle className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{safetyStatus?.total_violations || 0}</div>
            <p className="text-xs text-gray-600">
              {safetyStatus?.unresolved_violations || 0} unresolved
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Human Oversight</CardTitle>
            <Users className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{safetyStatus?.human_overseers || 0}</div>
            <p className="text-xs text-gray-600">
              {safetyStatus?.pending_approvals || 0} pending approvals
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Alignment Checks</CardTitle>
            <Eye className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{safetyStatus?.alignment_checks || 0}</div>
            <p className="text-xs text-gray-600">
              {safetyStatus?.last_alignment_check ? 
                `Last: ${new Date(safetyStatus.last_alignment_check).toLocaleTimeString()}` : 
                'No recent checks'
              }
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Middleware Statistics */}
      {middlewareStats && (
        <Card>
          <CardHeader>
            <CardTitle>Safety Middleware Statistics</CardTitle>
            <CardDescription>Operation filtering and safety enforcement</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-2xl font-bold">{middlewareStats.total_operations}</div>
                <p className="text-sm text-gray-600">Total Operations</p>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">{middlewareStats.success_rate.toFixed(1)}%</div>
                <p className="text-sm text-gray-600">Success Rate</p>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-600">{middlewareStats.block_rate.toFixed(1)}%</div>
                <p className="text-sm text-gray-600">Block Rate</p>
              </div>
              <div>
                <div className="text-2xl font-bold">{middlewareStats.blocked_operations}</div>
                <p className="text-sm text-gray-600">Blocked Operations</p>
              </div>
            </div>
            <div className="mt-4">
              <div className="flex justify-between text-sm mb-2">
                <span>Operation Success Rate</span>
                <span>{middlewareStats.success_rate.toFixed(1)}%</span>
              </div>
              <Progress value={middlewareStats.success_rate} className="h-2" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detailed Tabs */}
      <Tabs defaultValue="violations" className="space-y-4">
        <TabsList>
          <TabsTrigger value="violations">Safety Violations</TabsTrigger>
          <TabsTrigger value="alignment">Alignment Checks</TabsTrigger>
          <TabsTrigger value="overseers">Human Overseers</TabsTrigger>
          <TabsTrigger value="constraints">Constraints</TabsTrigger>
        </TabsList>

        <TabsContent value="violations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Safety Violations</CardTitle>
              <CardDescription>Critical safety constraint violations requiring attention</CardDescription>
            </CardHeader>
            <CardContent>
              {violations.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
                  <p>No safety violations detected</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {violations.slice(0, 10).map((violation) => (
                    <div key={violation.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Badge className={getSeverityColor(violation.severity)}>
                          {violation.severity.toUpperCase()}
                        </Badge>
                        <div className="flex items-center gap-2">
                          {violation.resolved ? (
                            <Badge variant="outline" className="text-green-600">Resolved</Badge>
                          ) : (
                            <Badge variant="destructive">Unresolved</Badge>
                          )}
                          <span className="text-sm text-gray-500">
                            {new Date(violation.timestamp).toLocaleString()}
                          </span>
                        </div>
                      </div>
                      <h4 className="font-medium mb-1">{violation.violation_type}</h4>
                      <p className="text-sm text-gray-600">{violation.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alignment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Alignment Verification</CardTitle>
              <CardDescription>Checks for alignment with human values and intentions</CardDescription>
            </CardHeader>
            <CardContent>
              {alignmentChecks.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Eye className="h-12 w-12 mx-auto mb-4" />
                  <p>No alignment checks performed yet</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {alignmentChecks.slice(0, 10).map((check) => (
                    <div key={check.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Badge className={getAlignmentColor(check.status)}>
                          {check.status.toUpperCase()}
                        </Badge>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">
                            {(check.confidence_score * 100).toFixed(1)}% confidence
                          </span>
                          <span className="text-sm text-gray-500">
                            {new Date(check.timestamp).toLocaleString()}
                          </span>
                        </div>
                      </div>
                      <h4 className="font-medium mb-1">{check.check_type}</h4>
                      <p className="text-sm text-gray-600">{check.description}</p>
                      <div className="mt-2">
                        <Progress value={check.confidence_score * 100} className="h-2" />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="overseers" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Human Oversight Personnel</CardTitle>
              <CardDescription>Authorized personnel for safety oversight and emergency response</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {overseers.map((overseer) => (
                  <div key={overseer.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">{overseer.name}</h4>
                      <div className="flex items-center gap-2">
                        <Badge variant={overseer.active ? "default" : "secondary"}>
                          {overseer.active ? "Active" : "Inactive"}
                        </Badge>
                        <Badge className={getSeverityColor(overseer.clearance_level)}>
                          {overseer.clearance_level.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{overseer.role}</p>
                    <div className="flex flex-wrap gap-1">
                      {overseer.permissions.map((permission) => (
                        <Badge key={permission} variant="outline" className="text-xs">
                          {permission}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Last active: {new Date(overseer.last_active).toLocaleString()}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="constraints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Ethical Constraints</CardTitle>
              <CardDescription>Active safety and ethical constraints governing AI behavior</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-gray-500">
                <Settings className="h-12 w-12 mx-auto mb-4" />
                <p>Constraint management interface coming soon</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}