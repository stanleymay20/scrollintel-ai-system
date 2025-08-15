"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  CreditCard, 
  Wallet, 
  Receipt, 
  AlertTriangle,
  Download,
  Eye,
  Plus,
  Trash2,
  Star,
  TrendingUp,
  DollarSign,
  Calendar
} from 'lucide-react';

interface ScrollCoinWallet {
  id: string;
  balance: number;
  reserved_balance: number;
  available_balance: number;
  last_transaction_at?: string;
  created_at: string;
}

interface Transaction {
  id: string;
  type: string;
  amount: number;
  balance_after: number;
  description?: string;
  created_at: string;
}

interface PaymentMethod {
  id: string;
  type: string;
  last_four?: string;
  brand?: string;
  exp_month?: number;
  exp_year?: number;
  is_default: boolean;
  nickname?: string;
  created_at: string;
}

interface Invoice {
  id: string;
  invoice_number: string;
  status: string;
  total_amount: number;
  currency: string;
  period_start: string;
  period_end: string;
  issued_at: string;
  due_date: string;
  paid_at?: string;
  is_overdue: boolean;
}

interface BillingAlert {
  id: string;
  type: string;
  severity: string;
  title: string;
  message: string;
  is_read: boolean;
  action_required: boolean;
  action_url?: string;
  action_text?: string;
  created_at: string;
}

interface UsageAnalytics {
  period_days: number;
  total_actions: number;
  total_cost: number;
  action_breakdown: Record<string, number>;
  cost_breakdown: Record<string, number>;
  daily_usage: Array<{ date: string; usage: number; cost: number }>;
  top_actions: Array<[string, number]>;
}

export default function BillingDashboard() {
  const [wallet, setWallet] = useState<ScrollCoinWallet | null>(null);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [alerts, setAlerts] = useState<BillingAlert[]>([]);
  const [usageAnalytics, setUsageAnalytics] = useState<UsageAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [rechargeAmount, setRechargeAmount] = useState<number>(50);
  const [showRechargeModal, setShowRechargeModal] = useState(false);

  useEffect(() => {
    fetchBillingData();
  }, []);

  const fetchBillingData = async () => {
    try {
      const [walletRes, paymentMethodsRes, invoicesRes, alertsRes, analyticsRes] = await Promise.all([
        fetch('/api/billing/wallet', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/billing/payment-methods', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/billing/invoices', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/billing/alerts', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/billing/usage/analytics', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        })
      ]);

      if (walletRes.ok) {
        const walletData = await walletRes.json();
        setWallet(walletData.wallet);
        setRecentTransactions(walletData.recent_transactions);
      }

      if (paymentMethodsRes.ok) {
        const pmData = await paymentMethodsRes.json();
        setPaymentMethods(pmData.payment_methods);
      }

      if (invoicesRes.ok) {
        const invoiceData = await invoicesRes.json();
        setInvoices(invoiceData.invoices);
      }

      if (alertsRes.ok) {
        const alertData = await alertsRes.json();
        setAlerts(alertData.alerts);
      }

      if (analyticsRes.ok) {
        const analyticsData = await analyticsRes.json();
        setUsageAnalytics(analyticsData);
      }
    } catch (error) {
      console.error('Error fetching billing data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRechargeWallet = async () => {
    if (!paymentMethods.length) {
      alert('Please add a payment method first');
      return;
    }

    const defaultPaymentMethod = paymentMethods.find(pm => pm.is_default) || paymentMethods[0];

    try {
      const response = await fetch('/api/billing/wallet/recharge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          amount: rechargeAmount,
          payment_method_id: defaultPaymentMethod.id
        })
      });

      if (response.ok) {
        await fetchBillingData();
        setShowRechargeModal(false);
        // Show success message
      }
    } catch (error) {
      console.error('Error recharging wallet:', error);
    }
  };

  const handleDeletePaymentMethod = async (paymentMethodId: string) => {
    if (!confirm('Are you sure you want to delete this payment method?')) {
      return;
    }

    try {
      const response = await fetch(`/api/billing/payment-methods/${paymentMethodId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        await fetchBillingData();
      }
    } catch (error) {
      console.error('Error deleting payment method:', error);
    }
  };

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount);
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'paid':
      case 'succeeded':
        return 'bg-green-100 text-green-800';
      case 'pending':
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
      case 'overdue':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
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
      {/* Billing Alerts */}
      {alerts.filter(alert => !alert.is_read).length > 0 && (
        <div className="space-y-2">
          {alerts.filter(alert => !alert.is_read).slice(0, 3).map((alert) => (
            <Alert key={alert.id} className={alert.severity === 'critical' ? 'border-red-500' : ''}>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <strong>{alert.title}</strong>: {alert.message}
                {alert.action_url && (
                  <Button variant="link" size="sm" className="p-0 ml-2">
                    {alert.action_text || 'Take Action'}
                  </Button>
                )}
              </AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ScrollCoin Balance</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{wallet?.balance.toLocaleString() || 0}</div>
            <p className="text-xs text-muted-foreground">
              {wallet?.reserved_balance || 0} reserved
            </p>
            <Button 
              size="sm" 
              className="mt-2 w-full"
              onClick={() => setShowRechargeModal(true)}
            >
              <Plus className="h-4 w-4 mr-2" />
              Recharge
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">This Month's Usage</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {usageAnalytics?.total_cost ? formatCurrency(usageAnalytics.total_cost) : '$0.00'}
            </div>
            <p className="text-xs text-muted-foreground">
              {usageAnalytics?.total_actions || 0} actions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Outstanding Invoices</CardTitle>
            <Receipt className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {invoices.filter(inv => inv.status !== 'paid').length}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatCurrency(
                invoices
                  .filter(inv => inv.status !== 'paid')
                  .reduce((sum, inv) => sum + inv.total_amount, 0)
              )} total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Payment Methods</CardTitle>
            <CreditCard className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{paymentMethods.length}</div>
            <p className="text-xs text-muted-foreground">
              {paymentMethods.filter(pm => pm.is_default).length} default
            </p>
            <Button size="sm" variant="outline" className="mt-2 w-full">
              <Plus className="h-4 w-4 mr-2" />
              Add New
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="wallet" className="space-y-4">
        <TabsList>
          <TabsTrigger value="wallet">ScrollCoin Wallet</TabsTrigger>
          <TabsTrigger value="invoices">Invoices</TabsTrigger>
          <TabsTrigger value="payment-methods">Payment Methods</TabsTrigger>
          <TabsTrigger value="analytics">Usage Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="wallet" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>ScrollCoin Wallet</CardTitle>
              <CardDescription>
                Manage your ScrollCoin balance and view transaction history
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
                  <div>
                    <p className="text-sm text-gray-500">Available Balance</p>
                    <p className="text-2xl font-bold text-green-600">
                      {wallet?.available_balance.toLocaleString() || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Reserved</p>
                    <p className="text-lg font-semibold text-yellow-600">
                      {wallet?.reserved_balance.toLocaleString() || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Total Balance</p>
                    <p className="text-lg font-semibold">
                      {wallet?.balance.toLocaleString() || 0}
                    </p>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Recent Transactions</h4>
                  <div className="space-y-2">
                    {recentTransactions.map((transaction) => (
                      <div key={transaction.id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <p className="font-medium capitalize">{transaction.type.replace('_', ' ')}</p>
                          <p className="text-sm text-gray-500">
                            {new Date(transaction.created_at).toLocaleDateString()}
                          </p>
                          {transaction.description && (
                            <p className="text-sm text-gray-600">{transaction.description}</p>
                          )}
                        </div>
                        <div className="text-right">
                          <p className={`font-semibold ${
                            transaction.amount > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {transaction.amount > 0 ? '+' : ''}{transaction.amount.toLocaleString()}
                          </p>
                          <p className="text-sm text-gray-500">
                            Balance: {transaction.balance_after.toLocaleString()}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="invoices" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Billing Invoices</CardTitle>
              <CardDescription>
                View and download your billing invoices
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {invoices.map((invoice) => (
                  <div key={invoice.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <p className="font-medium">Invoice #{invoice.invoice_number}</p>
                      <p className="text-sm text-gray-500">
                        {new Date(invoice.period_start).toLocaleDateString()} - {new Date(invoice.period_end).toLocaleDateString()}
                      </p>
                      <Badge className={getStatusColor(invoice.status)}>
                        {invoice.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">{formatCurrency(invoice.total_amount, invoice.currency)}</p>
                      <p className="text-sm text-gray-500">
                        Due: {new Date(invoice.due_date).toLocaleDateString()}
                      </p>
                      <div className="flex gap-2 mt-2">
                        <Button size="sm" variant="outline">
                          <Eye className="h-4 w-4 mr-2" />
                          View
                        </Button>
                        <Button size="sm" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="payment-methods" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Payment Methods</CardTitle>
              <CardDescription>
                Manage your saved payment methods
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {paymentMethods.map((method) => (
                  <div key={method.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <CreditCard className="h-8 w-8 text-gray-400" />
                      <div>
                        <p className="font-medium">
                          {method.brand?.toUpperCase()} •••• {method.last_four}
                          {method.is_default && (
                            <Badge variant="secondary" className="ml-2">Default</Badge>
                          )}
                        </p>
                        <p className="text-sm text-gray-500">
                          Expires {method.exp_month}/{method.exp_year}
                        </p>
                        {method.nickname && (
                          <p className="text-sm text-gray-600">{method.nickname}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {!method.is_default && (
                        <Button size="sm" variant="outline">
                          <Star className="h-4 w-4 mr-2" />
                          Set Default
                        </Button>
                      )}
                      <Button 
                        size="sm" 
                        variant="destructive"
                        onClick={() => handleDeletePaymentMethod(method.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Usage Analytics</CardTitle>
              <CardDescription>
                Detailed breakdown of your ScrollIntel usage
              </CardDescription>
            </CardHeader>
            <CardContent>
              {usageAnalytics && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <p className="text-2xl font-bold text-blue-600">
                        {usageAnalytics.total_actions.toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-600">Total Actions</p>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <p className="text-2xl font-bold text-green-600">
                        {formatCurrency(usageAnalytics.total_cost)}
                      </p>
                      <p className="text-sm text-gray-600">Total Cost</p>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <p className="text-2xl font-bold text-purple-600">
                        {usageAnalytics.period_days}
                      </p>
                      <p className="text-sm text-gray-600">Days Period</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Top Actions</h4>
                    <div className="space-y-2">
                      {usageAnalytics.top_actions.map(([action, count], index) => (
                        <div key={action} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="capitalize">{action.replace('_', ' ')}</span>
                          <div className="text-right">
                            <span className="font-semibold">{count.toLocaleString()}</span>
                            <span className="text-sm text-gray-500 ml-2">
                              {formatCurrency(usageAnalytics.cost_breakdown[action] || 0)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Recharge Modal */}
      {showRechargeModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Recharge ScrollCoin Wallet</CardTitle>
              <CardDescription>
                Add ScrollCoins to your wallet (1 USD = 100 ScrollCoins)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Amount (USD)</label>
                  <input
                    type="number"
                    value={rechargeAmount}
                    onChange={(e) => setRechargeAmount(Number(e.target.value))}
                    className="w-full p-2 border rounded-lg"
                    min="10"
                    max="1000"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    You will receive {(rechargeAmount * 100).toLocaleString()} ScrollCoins
                  </p>
                </div>
                
                <div className="flex gap-2">
                  <Button onClick={handleRechargeWallet} className="flex-1">
                    <DollarSign className="h-4 w-4 mr-2" />
                    Recharge {formatCurrency(rechargeAmount)}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => setShowRechargeModal(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}