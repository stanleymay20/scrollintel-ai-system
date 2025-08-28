'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Check, Star, Zap, Crown } from 'lucide-react'

interface SubscriptionPlan {
  id: string
  name: string
  description: string
  price: number
  currency: string
  interval: 'monthly' | 'yearly'
  features: string[]
  popular?: boolean
  current?: boolean
  stripe_price_id?: string
  paypal_plan_id?: string
}

interface SubscriptionPlansProps {
  onSelectPlan?: (plan: SubscriptionPlan) => void
  currentPlanId?: string
}

export function SubscriptionPlans({ onSelectPlan, currentPlanId }: SubscriptionPlansProps) {
  const [billingInterval, setBillingInterval] = useState<'monthly' | 'yearly'>('monthly')
  const [loading, setLoading] = useState<string | null>(null)

  const plans: SubscriptionPlan[] = [
    {
      id: 'free',
      name: 'Free',
      description: 'Perfect for getting started',
      price: 0,
      currency: 'USD',
      interval: billingInterval,
      features: [
        '5 AI agent interactions per day',
        'Basic analytics dashboard',
        'Community support',
        '1 workspace',
        'Basic file processing'
      ]
    },
    {
      id: 'pro',
      name: 'Pro',
      description: 'For professionals and small teams',
      price: billingInterval === 'monthly' ? 29 : 290,
      currency: 'USD',
      interval: billingInterval,
      popular: true,
      stripe_price_id: billingInterval === 'monthly' ? 'price_pro_monthly' : 'price_pro_yearly',
      paypal_plan_id: billingInterval === 'monthly' ? 'plan_pro_monthly' : 'plan_pro_yearly',
      features: [
        'Unlimited AI agent interactions',
        'Advanced analytics & insights',
        'Priority support',
        '10 workspaces',
        'Advanced file processing',
        'Custom integrations',
        'API access',
        'Export capabilities'
      ]
    },
    {
      id: 'enterprise',
      name: 'Enterprise',
      description: 'For large organizations',
      price: billingInterval === 'monthly' ? 99 : 990,
      currency: 'USD',
      interval: billingInterval,
      stripe_price_id: billingInterval === 'monthly' ? 'price_enterprise_monthly' : 'price_enterprise_yearly',
      paypal_plan_id: billingInterval === 'monthly' ? 'plan_enterprise_monthly' : 'plan_enterprise_yearly',
      features: [
        'Everything in Pro',
        'Unlimited workspaces',
        'Advanced security features',
        'Dedicated account manager',
        'Custom AI model training',
        'On-premise deployment',
        'SLA guarantee',
        'Advanced compliance features'
      ]
    }
  ]

  const handleSelectPlan = async (plan: SubscriptionPlan) => {
    if (plan.id === currentPlanId) return
    
    setLoading(plan.id)
    try {
      onSelectPlan?.(plan)
    } catch (error) {
      console.error('Failed to select plan:', error)
    } finally {
      setLoading(null)
    }
  }

  const getPlanIcon = (planId: string) => {
    switch (planId) {
      case 'free':
        return <Star className="h-6 w-6" />
      case 'pro':
        return <Zap className="h-6 w-6" />
      case 'enterprise':
        return <Crown className="h-6 w-6" />
      default:
        return <Star className="h-6 w-6" />
    }
  }

  const formatPrice = (price: number, currency: string, interval: string) => {
    if (price === 0) return 'Free'
    
    const formattedPrice = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(price)
    
    return `${formattedPrice}/${interval === 'yearly' ? 'year' : 'month'}`
  }

  const getYearlySavings = (monthlyPrice: number) => {
    const yearlyPrice = monthlyPrice * 10 // 2 months free
    const monthlySavings = (monthlyPrice * 12) - yearlyPrice
    return Math.round((monthlySavings / (monthlyPrice * 12)) * 100)
  }

  return (
    <div className="space-y-6">
      {/* Billing Toggle */}
      <div className="flex justify-center">
        <div className="bg-gray-100 p-1 rounded-lg">
          <button
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              billingInterval === 'monthly'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            onClick={() => setBillingInterval('monthly')}
          >
            Monthly
          </button>
          <button
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              billingInterval === 'yearly'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            onClick={() => setBillingInterval('yearly')}
          >
            Yearly
            <Badge variant="secondary" className="ml-2">
              Save 17%
            </Badge>
          </button>
        </div>
      </div>

      {/* Plans Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {plans.map((plan) => (
          <Card
            key={`${plan.id}-${billingInterval}`}
            className={`relative ${
              plan.popular ? 'border-blue-500 shadow-lg' : ''
            } ${
              plan.id === currentPlanId ? 'ring-2 ring-green-500' : ''
            }`}
          >
            {plan.popular && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <Badge className="bg-blue-500 text-white">Most Popular</Badge>
              </div>
            )}
            
            {plan.id === currentPlanId && (
              <div className="absolute -top-3 right-4">
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  Current Plan
                </Badge>
              </div>
            )}

            <CardHeader className="text-center">
              <div className="flex justify-center mb-4">
                {getPlanIcon(plan.id)}
              </div>
              <CardTitle className="text-xl">{plan.name}</CardTitle>
              <CardDescription>{plan.description}</CardDescription>
              <div className="mt-4">
                <div className="text-3xl font-bold">
                  {formatPrice(plan.price, plan.currency, plan.interval)}
                </div>
                {billingInterval === 'yearly' && plan.price > 0 && (
                  <div className="text-sm text-gray-500 mt-1">
                    Save {getYearlySavings(plan.price / 10)}% vs monthly
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <ul className="space-y-2">
                {plan.features.map((feature, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <Check className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button
                className="w-full"
                variant={plan.id === currentPlanId ? 'outline' : 'default'}
                disabled={plan.id === currentPlanId || loading === plan.id}
                onClick={() => handleSelectPlan(plan)}
              >
                {loading === plan.id ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Processing...</span>
                  </div>
                ) : plan.id === currentPlanId ? (
                  'Current Plan'
                ) : plan.price === 0 ? (
                  'Get Started'
                ) : (
                  `Upgrade to ${plan.name}`
                )}
              </Button>

              {plan.price > 0 && (
                <div className="text-xs text-gray-500 text-center">
                  Cancel anytime. No hidden fees.
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Enterprise Contact */}
      <div className="text-center py-8">
        <h3 className="text-lg font-medium mb-2">Need a custom solution?</h3>
        <p className="text-gray-600 mb-4">
          Contact our sales team for enterprise pricing and custom features.
        </p>
        <Button variant="outline">
          Contact Sales
        </Button>
      </div>
    </div>
  )
}