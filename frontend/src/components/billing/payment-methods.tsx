'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { CreditCard, Plus, Trash2, CheckCircle, AlertCircle } from 'lucide-react'

interface PaymentMethod {
  id: string
  type: 'card' | 'paypal' | 'bank_account'
  brand?: string
  last_four?: string
  exp_month?: number
  exp_year?: number
  is_default: boolean
  is_active: boolean
}

interface PaymentMethodsProps {
  onAddPaymentMethod?: () => void
  onDeletePaymentMethod?: (id: string) => void
  onSetDefault?: (id: string) => void
}

export function PaymentMethods({ 
  onAddPaymentMethod, 
  onDeletePaymentMethod, 
  onSetDefault 
}: PaymentMethodsProps) {
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPaymentMethods()
  }, [])

  const fetchPaymentMethods = async () => {
    try {
      setLoading(true)
      // In production, this would be an actual API call
      const mockData: PaymentMethod[] = [
        {
          id: 'pm_1',
          type: 'card',
          brand: 'visa',
          last_four: '4242',
          exp_month: 12,
          exp_year: 2025,
          is_default: true,
          is_active: true
        },
        {
          id: 'pm_2',
          type: 'paypal',
          is_default: false,
          is_active: true
        }
      ]
      
      setPaymentMethods(mockData)
      setError(null)
    } catch (err) {
      setError('Failed to load payment methods')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      // In production, this would be an actual API call
      setPaymentMethods(prev => prev.filter(pm => pm.id !== id))
      onDeletePaymentMethod?.(id)
    } catch (err) {
      setError('Failed to delete payment method')
    }
  }

  const handleSetDefault = async (id: string) => {
    try {
      // In production, this would be an actual API call
      setPaymentMethods(prev => 
        prev.map(pm => ({
          ...pm,
          is_default: pm.id === id
        }))
      )
      onSetDefault?.(id)
    } catch (err) {
      setError('Failed to set default payment method')
    }
  }

  const getPaymentMethodIcon = (type: string, brand?: string) => {
    if (type === 'card') {
      return <CreditCard className="h-5 w-5" />
    }
    return <CreditCard className="h-5 w-5" />
  }

  const getPaymentMethodDisplay = (method: PaymentMethod) => {
    if (method.type === 'card') {
      return `${method.brand?.toUpperCase()} •••• ${method.last_four}`
    } else if (method.type === 'paypal') {
      return 'PayPal Account'
    } else if (method.type === 'bank_account') {
      return `Bank Account •••• ${method.last_four}`
    }
    return 'Unknown Payment Method'
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Payment Methods</CardTitle>
          <CardDescription>Manage your payment methods</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Payment Methods</CardTitle>
        <CardDescription>
          Manage your payment methods for subscriptions and purchases
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {paymentMethods.length === 0 ? (
          <div className="text-center py-8">
            <CreditCard className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No payment methods
            </h3>
            <p className="text-gray-500 mb-4">
              Add a payment method to start using ScrollIntel services
            </p>
            <Button onClick={onAddPaymentMethod}>
              <Plus className="h-4 w-4 mr-2" />
              Add Payment Method
            </Button>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {paymentMethods.map((method) => (
                <div
                  key={method.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    {getPaymentMethodIcon(method.type, method.brand)}
                    <div>
                      <div className="font-medium">
                        {getPaymentMethodDisplay(method)}
                      </div>
                      {method.type === 'card' && method.exp_month && method.exp_year && (
                        <div className="text-sm text-gray-500">
                          Expires {method.exp_month.toString().padStart(2, '0')}/{method.exp_year}
                        </div>
                      )}
                    </div>
                    {method.is_default && (
                      <Badge variant="secondary">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Default
                      </Badge>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {!method.is_default && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleSetDefault(method.id)}
                      >
                        Set Default
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDelete(method.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            <div className="pt-4 border-t">
              <Button onClick={onAddPaymentMethod} variant="outline" className="w-full">
                <Plus className="h-4 w-4 mr-2" />
                Add New Payment Method
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}