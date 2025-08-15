"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  CreditCard, 
  Crown, 
  Star, 
  Building, 
  Zap,
  CheckCircle,
  AlertCircle,
  Calendar,
  DollarSign,
  TrendingUp
} from 'lucide-react';

interface SubscriptionTier {
  id: string;
  name: string;
  price: {
    monthly: number;
    yearly: number;
  };
  features: string[];
  limits: {
    api_calls: number;
    training_jobs: number;
    storage_gb: number;
    scrollcoins: number;
  };
  popular?: boolean;
  icon: React.ReactNode;
}

interface CurrentSubscription {
  id: string;
  tier: string;
  status: string;
  billing_cycle: string;
  base_price: number;
  currency: string;
  current_period_start: string;
  current_period_end: string;
  next_billing_date?: string;
  trial_end?: string;
  is_trial: boolean;
  days_until_renewal?: number;
}

interface UsageStats {
  current_period: {
    api_calls: number;
    training_jobs: number;
    storage_gb: number;
    scrollcoins_used: number;
  };
}

const SUBSCRIPTION_TIERS: SubscriptionTier[] = [
  {
    id: 'free',
    name: 'Free',
    price: { monthly: 0, yearly: 0 },
    features: [
      'Basic data analysis',
      'Simple visualizations',
      'Community support',
      '1 GB storage'
    ],
    limits: {
      api_calls: 1000,
      training_jobs: 1,
      storage_gb: 1,
      scrollcoins: 100
    },
    icon: <Star className="h-6 w-6" />
  },
  {
    id: 'starter',
    name: 'Starter',
    price: { monthly: 29, yearly: 290 },
    features: [
      'Advanced analysis',
      'ML model training',
      'Basic explanations',
      'Email support',
      '10 GB storage'
    ],
    limits: {
      api_calls: 10000,
      training_jobs: 10,
      storage_gb: 10,
      scrollcoins: 1000
    },
    popular: true,
    icon: <Zap className="h-6 w-6" />
  },
  {
    id: 'professional',
    name: 'Professional',
    price: { monthly: 99, yearly: 990 },
    features: [
      'Full ML suite',
      'Advanced explanations',
      'Multimodal AI',
      'Priority support',
      '100 GB storage'
    ],
    limits: {
      api_calls: 100000,
      training_jobs: 50,
      storage_gb: 100,
      scrollcoins: 5000
    },
    icon: <Crown className="h-6 w-6" />
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: { monthly: 299, yearly: 2990 },
    features: [
      'Enterprise features',
      'Custom models',
      'Compliance tools',
      'Dedicated support',
      '1 TB storage'
    ],
    limits: {
      api_calls: 1000000,
      training_jobs: 200,
      storage_gb: 1000,
      scrollcoins: 20000
    },
    icon: <Building className="h-6 w-6" />
  }
];

export default function SubscriptionManager() {
  const [currentSubscription, setCurrentSubscription] = useState<CurrentSubscription | null>(null);
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [tierLimits, setTierLimits] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [upgrading, setUpgrading] = useState(false);
  const [selectedTier, setSelectedTier] = useState<string>('');
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'yearly'>('monthly');

  useEffect(() => {
    fetchSubscriptionData();
  }, []);

  const fetchSubscriptionData = async () => {
    try {
      const response = await fetch('/api/billing/subscription', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentSubscription(data.subscription);
        setUsageStats(data.usage_stats);
        setTierLimits(data.tier_limits);
      }
    } catch (error) {
      console.error('Error fetching subscription data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgrade = async (tierId: string) => {
    setUpgrading(true);
    setSelectedTier(tierId);

    try {
      const response = await fetch('/api/billing/subscription', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          tier: tierId,
          billing_cycle: billingCycle
        })
      });

      if (response.ok) {
        const data = await response.json();
        await fetchSubscriptionData();
        // Show success message
      } else {
        // Handle error
        console.error('Upgrade failed');
      }
    } catch (error) {
      console.error('Error upgrading subscription:', error);
    } finally {
      setUpgrading(false);
      setSelectedTier('');
    }
  };

  const handleCancelSubscription = async () => {
    if (!confirm('Are you sure you want to cancel your subscription?')) {
      return;
    }

    try {
      const response = await fetch('/api/billing/subscription', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        await fetchSubscriptionData();
        // Show success message
      }
    } catch (error) {
      console.error('Error cancelling subscription:', error);
    }
  };

  const formatUsagePercentage = (used: number, limit: number): number => {
    if (limit === -1) return 0; // Unlimited
    return Math.min((used / limit) * 100, 100);
  };

  const getUsageColor = (percentage: number): string => {
    if (percentage >= 90) return 'bg-red-500';
    if (percentage >= 75) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current Subscription Overview */}
      {currentSubscription && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CreditCard className="h-5 w-5" />
              Current Subscription
            </CardTitle>
            <CardDescription>
              Manage your ScrollIntel subscription and usage
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant={currentSubscription.status === 'active' ? 'default' : 'secondary'}>
                    {currentSubscription.tier.toUpperCase()}
                  </Badge>
                  {currentSubscription.is_trial && (
                    <Badge variant="outline">Trial</Badge>
                  )}
                </div>
                <p className="text-2xl font-bold">
                  ${currentSubscription.base_price}
                  <span className="text-sm font-normal text-gray-500">
                    /{currentSubscription.billing_cycle}
                  </span>
                </p>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 mb-1">Next billing date</p>
                <p className="font-medium">
                  {currentSubscription.next_billing_date 
                    ? new Date(currentSubscription.next_billing_date).toLocaleDateString()
                    : 'N/A'
                  }
                </p>
                {currentSubscription.days_until_renewal && (
                  <p className="text-sm text-gray-500">
                    {currentSubscription.days_until_renewal} days remaining
                  </p>
                )}
              </div>

              <div className="flex items-center gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => window.open('/billing/invoices', '_blank')}
                >
                  <Calendar className="h-4 w-4 mr-2" />
                  View Invoices
                </Button>
                {currentSubscription.tier !== 'free' && (
                  <Button 
                    variant="destructive" 
                    size="sm"
                    onClick={handleCancelSubscription}
                  >
                    Cancel
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Usage Statistics */}
      {usageStats && tierLimits && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Usage This Period
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">API Calls</span>
                  <span className="text-sm text-gray-500">
                    {usageStats.current_period.api_calls.toLocaleString()} / {
                      tierLimits.api_calls === -1 ? '∞' : tierLimits.api_calls.toLocaleString()
                    }
                  </span>
                </div>
                <Progress 
                  value={formatUsagePercentage(usageStats.current_period.api_calls, tierLimits.api_calls)}
                  className="h-2"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Training Jobs</span>
                  <span className="text-sm text-gray-500">
                    {usageStats.current_period.training_jobs} / {
                      tierLimits.training_jobs === -1 ? '∞' : tierLimits.training_jobs
                    }
                  </span>
                </div>
                <Progress 
                  value={formatUsagePercentage(usageStats.current_period.training_jobs, tierLimits.training_jobs)}
                  className="h-2"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Storage</span>
                  <span className="text-sm text-gray-500">
                    {usageStats.current_period.storage_gb.toFixed(1)} GB / {
                      tierLimits.storage_gb === -1 ? '∞' : `${tierLimits.storage_gb} GB`
                    }
                  </span>
                </div>
                <Progress 
                  value={formatUsagePercentage(usageStats.current_period.storage_gb, tierLimits.storage_gb)}
                  className="h-2"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">ScrollCoins</span>
                  <span className="text-sm text-gray-500">
                    {usageStats.current_period.scrollcoins_used} used
                  </span>
                </div>
                <Button variant="outline" size="sm" className="w-full">
                  <DollarSign className="h-4 w-4 mr-2" />
                  Recharge Wallet
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Subscription Plans */}
      <Card>
        <CardHeader>
          <CardTitle>Subscription Plans</CardTitle>
          <CardDescription>
            Choose the plan that best fits your needs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={billingCycle} onValueChange={(value) => setBillingCycle(value as 'monthly' | 'yearly')}>
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="monthly">Monthly</TabsTrigger>
              <TabsTrigger value="yearly">
                Yearly
                <Badge variant="secondary" className="ml-2">Save 17%</Badge>
              </TabsTrigger>
            </TabsList>

            <TabsContent value={billingCycle}>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {SUBSCRIPTION_TIERS.map((tier) => (
                  <Card 
                    key={tier.id} 
                    className={`relative ${tier.popular ? 'ring-2 ring-blue-500' : ''} ${
                      currentSubscription?.tier === tier.id ? 'bg-blue-50 border-blue-200' : ''
                    }`}
                  >
                    {tier.popular && (
                      <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                        <Badge className="bg-blue-500">Most Popular</Badge>
                      </div>
                    )}
                    
                    <CardHeader className="text-center">
                      <div className="flex justify-center mb-2">
                        {tier.icon}
                      </div>
                      <CardTitle className="text-lg">{tier.name}</CardTitle>
                      <div className="text-3xl font-bold">
                        ${tier.price[billingCycle]}
                        <span className="text-sm font-normal text-gray-500">
                          /{billingCycle === 'yearly' ? 'year' : 'month'}
                        </span>
                      </div>
                      {billingCycle === 'yearly' && tier.price.yearly > 0 && (
                        <p className="text-sm text-green-600">
                          Save ${(tier.price.monthly * 12) - tier.price.yearly}/year
                        </p>
                      )}
                    </CardHeader>
                    
                    <CardContent>
                      <ul className="space-y-2 mb-4">
                        {tier.features.map((feature, index) => (
                          <li key={index} className="flex items-center gap-2 text-sm">
                            <CheckCircle className="h-4 w-4 text-green-500" />
                            {feature}
                          </li>
                        ))}
                      </ul>

                      <div className="space-y-1 text-xs text-gray-500 mb-4">
                        <p>{tier.limits.api_calls === -1 ? 'Unlimited' : tier.limits.api_calls.toLocaleString()} API calls</p>
                        <p>{tier.limits.training_jobs === -1 ? 'Unlimited' : tier.limits.training_jobs} training jobs</p>
                        <p>{tier.limits.storage_gb === -1 ? 'Unlimited' : `${tier.limits.storage_gb} GB`} storage</p>
                        <p>{tier.limits.scrollcoins.toLocaleString()} ScrollCoins included</p>
                      </div>

                      {currentSubscription?.tier === tier.id ? (
                        <Button disabled className="w-full">
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Current Plan
                        </Button>
                      ) : (
                        <Button 
                          className="w-full"
                          variant={tier.popular ? 'default' : 'outline'}
                          onClick={() => handleUpgrade(tier.id)}
                          disabled={upgrading && selectedTier === tier.id}
                        >
                          {upgrading && selectedTier === tier.id ? (
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          ) : null}
                          {tier.id === 'free' ? 'Downgrade' : 'Upgrade'}
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Billing Alerts */}
      {currentSubscription?.is_trial && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Your trial expires on {new Date(currentSubscription.trial_end!).toLocaleDateString()}. 
            Upgrade now to continue using ScrollIntel without interruption.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}