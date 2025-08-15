#!/usr/bin/env python3
"""
ScrollInsightRadar Demo Script

Demonstrates the comprehensive pattern detection, trend analysis, 
anomaly detection, and insight generation capabilities of ScrollInsightRadar.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from scrollintel.engines.scroll_insight_radar import ScrollInsightRadar
from scrollintel.models.schemas import PatternDetectionConfig


def create_sample_datasets():
    """Create various sample datasets for demonstration."""
    np.random.seed(42)
    
    datasets = {}
    
    # 1. Sales and Marketing Dataset with Correlations
    print("📊 Creating Sales & Marketing Dataset...")
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Base sales with trend and seasonality
    trend = np.linspace(1000, 1500, 365)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly seasonality
    weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
    noise = np.random.normal(0, 50, 365)
    
    sales = trend + seasonal + weekly_pattern + noise
    
    # Marketing spend correlated with sales
    marketing_spend = sales * 0.3 + np.random.normal(100, 30, 365)
    
    # Temperature (external factor)
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 3, 365)
    
    # Customer satisfaction (inversely related to temperature extremes)
    satisfaction = 8.5 - 0.1 * np.abs(temperature - 20) + np.random.normal(0, 0.5, 365)
    satisfaction = np.clip(satisfaction, 1, 10)
    
    datasets['sales_marketing'] = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'marketing_spend': marketing_spend,
        'temperature': temperature,
        'customer_satisfaction': satisfaction,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    # 2. Financial Time Series with Anomalies
    print("💰 Creating Financial Dataset with Anomalies...")
    financial_dates = pd.date_range('2023-01-01', periods=252, freq='B')  # Business days
    
    # Stock price simulation
    returns = np.random.normal(0.001, 0.02, 252)
    
    # Add some anomalous events
    anomaly_days = [50, 100, 150, 200]
    for day in anomaly_days:
        returns[day] = np.random.choice([-0.15, 0.12])  # Major drops or gains
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Volume with correlation to price changes
    volume = 1000000 + 500000 * np.abs(returns) + np.random.normal(0, 100000, 252)
    volume = np.maximum(volume, 100000)  # Minimum volume
    
    # Volatility
    volatility = pd.Series(returns).rolling(20).std() * np.sqrt(252)
    volatility = volatility.fillna(volatility.mean())
    
    datasets['financial'] = pd.DataFrame({
        'date': financial_dates,
        'stock_price': prices,
        'daily_return': returns,
        'volume': volume,
        'volatility': volatility,
        'high_volume_day': (volume > np.quantile(volume, 0.8)).astype(int)
    })
    
    # 3. E-commerce Dataset with Clustering Patterns
    print("🛒 Creating E-commerce Dataset...")
    n_customers = 1000
    
    # Customer segments
    segments = np.random.choice(['Budget', 'Premium', 'Luxury'], n_customers, p=[0.5, 0.3, 0.2])
    
    # Generate customer behavior based on segments
    customer_data = []
    for i, segment in enumerate(segments):
        if segment == 'Budget':
            avg_order_value = np.random.normal(25, 8)
            orders_per_month = np.random.poisson(2)
            session_duration = np.random.normal(5, 2)
        elif segment == 'Premium':
            avg_order_value = np.random.normal(75, 15)
            orders_per_month = np.random.poisson(4)
            session_duration = np.random.normal(12, 3)
        else:  # Luxury
            avg_order_value = np.random.normal(200, 50)
            orders_per_month = np.random.poisson(6)
            session_duration = np.random.normal(20, 5)
        
        customer_data.append({
            'customer_id': f'CUST_{i:04d}',
            'segment': segment,
            'avg_order_value': max(avg_order_value, 5),
            'orders_per_month': max(orders_per_month, 0),
            'session_duration_minutes': max(session_duration, 1),
            'total_spent': max(avg_order_value, 5) * max(orders_per_month, 0),
            'days_since_last_order': np.random.exponential(15),
            'email_open_rate': np.random.beta(2, 3),
            'mobile_user': np.random.choice([0, 1], p=[0.3, 0.7])
        })
    
    datasets['ecommerce'] = pd.DataFrame(customer_data)
    
    # 4. IoT Sensor Data with Multiple Patterns
    print("🌡️ Creating IoT Sensor Dataset...")
    sensor_dates = pd.date_range('2023-01-01', periods=2000, freq='H')
    
    # Multiple sensors with different patterns
    # Temperature sensor with daily cycle
    temp_base = 20 + 5 * np.sin(2 * np.pi * np.arange(2000) / 24)  # Daily cycle
    temp_noise = np.random.normal(0, 1, 2000)
    temperature_sensor = temp_base + temp_noise
    
    # Humidity sensor (inversely related to temperature)
    humidity = 60 - 0.5 * (temperature_sensor - 20) + np.random.normal(0, 3, 2000)
    humidity = np.clip(humidity, 20, 90)
    
    # Pressure sensor with weather patterns
    pressure = 1013 + 10 * np.sin(2 * np.pi * np.arange(2000) / 168) + np.random.normal(0, 2, 2000)  # Weekly pattern
    
    # Add some sensor malfunctions (anomalies)
    malfunction_indices = np.random.choice(2000, 20, replace=False)
    temperature_sensor[malfunction_indices] = np.random.uniform(-10, 50, 20)
    
    datasets['iot_sensors'] = pd.DataFrame({
        'timestamp': sensor_dates,
        'temperature': temperature_sensor,
        'humidity': humidity,
        'pressure': pressure,
        'sensor_status': np.where(np.isin(np.arange(2000), malfunction_indices), 'ERROR', 'OK'),
        'hour_of_day': sensor_dates.hour,
        'day_of_week': sensor_dates.dayofweek
    })
    
    return datasets


async def demonstrate_pattern_detection(radar, datasets):
    """Demonstrate pattern detection capabilities."""
    print("\n" + "="*60)
    print("🔍 PATTERN DETECTION DEMONSTRATION")
    print("="*60)
    
    for name, data in datasets.items():
        print(f"\n📈 Analyzing {name.replace('_', ' ').title()} Dataset...")
        print(f"   Dataset shape: {data.shape}")
        
        # Configure analysis based on dataset type
        if name == 'financial':
            config = PatternDetectionConfig(
                correlation_threshold=0.6,
                anomaly_contamination=0.05,  # Expect fewer anomalies in financial data
                significance_level=0.01
            )
        elif name == 'iot_sensors':
            config = PatternDetectionConfig(
                correlation_threshold=0.7,
                anomaly_contamination=0.02,  # Very few sensor malfunctions
                seasonal_period=24  # Hourly data with daily patterns
            )
        else:
            config = PatternDetectionConfig()
        
        try:
            results = await radar.detect_patterns(data, config)
            
            print(f"   ✅ Analysis completed successfully")
            print(f"   📊 Business Impact Score: {results['business_impact_score']:.3f}")
            print(f"   🔍 Insights Generated: {len(results['insights'])}")
            
            # Show top insights
            if results['insights']:
                print(f"   🏆 Top Insight: {results['insights'][0]['title']}")
                print(f"      Impact: {results['insights'][0]['impact_score']:.3f}")
                print(f"      Priority: {results['insights'][0]['priority']}")
            
            # Show pattern summary
            patterns = results['patterns']
            if 'correlation_patterns' in patterns and 'strong_correlations' in patterns['correlation_patterns']:
                corr_count = patterns['correlation_patterns']['total_correlations_found']
                print(f"   🔗 Strong Correlations Found: {corr_count}")
            
            if 'clustering_patterns' in patterns and 'clusters_found' in patterns['clustering_patterns']:
                cluster_count = patterns['clustering_patterns']['clusters_found']
                print(f"   🎯 Data Clusters Identified: {cluster_count}")
            
            # Show anomaly summary
            anomalies = results['anomalies']
            if 'total_anomalies_found' in anomalies:
                anomaly_count = anomalies['total_anomalies_found']
                anomaly_pct = anomalies.get('anomaly_percentage', 0)
                print(f"   ⚠️  Anomalies Detected: {anomaly_count} ({anomaly_pct:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {str(e)}")


async def demonstrate_trend_analysis(radar, datasets):
    """Demonstrate trend analysis capabilities."""
    print("\n" + "="*60)
    print("📈 TREND ANALYSIS DEMONSTRATION")
    print("="*60)
    
    time_series_datasets = ['sales_marketing', 'financial', 'iot_sensors']
    
    for name in time_series_datasets:
        if name not in datasets:
            continue
            
        data = datasets[name]
        print(f"\n📊 Trend Analysis for {name.replace('_', ' ').title()}...")
        
        try:
            config = PatternDetectionConfig(significance_level=0.05)
            trends = await radar._analyze_trends(data, config)
            
            if 'trend_analysis' in trends:
                significant_trends = [t for t in trends['trend_analysis'] if t['is_significant']]
                print(f"   📈 Significant Trends Found: {len(significant_trends)}")
                
                for trend in significant_trends[:3]:  # Show top 3
                    direction = "📈" if trend['trend_direction'] == 'increasing' else "📉"
                    print(f"   {direction} {trend['numeric_column']}: {trend['trend_direction']} "
                          f"(R² = {trend['r_squared']:.3f}, p = {trend['p_value']:.4f})")
                    print(f"      Confidence: {trend['confidence_level']}, "
                          f"Strength: {trend['trend_strength']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Trend analysis failed: {str(e)}")


async def demonstrate_anomaly_detection(radar, datasets):
    """Demonstrate anomaly detection capabilities."""
    print("\n" + "="*60)
    print("⚠️  ANOMALY DETECTION DEMONSTRATION")
    print("="*60)
    
    for name, data in datasets.items():
        print(f"\n🔍 Anomaly Detection for {name.replace('_', ' ').title()}...")
        
        try:
            config = PatternDetectionConfig(
                anomaly_contamination=0.1 if name != 'iot_sensors' else 0.02
            )
            anomalies = await radar._detect_anomalies(data, config)
            
            if 'total_anomalies_found' in anomalies:
                total_anomalies = anomalies['total_anomalies_found']
                anomaly_pct = anomalies.get('anomaly_percentage', 0)
                print(f"   ⚠️  Total Anomalies: {total_anomalies} ({anomaly_pct:.1f}%)")
                
                # Show high severity anomalies
                if 'isolation_forest_anomalies' in anomalies:
                    high_severity = [a for a in anomalies['isolation_forest_anomalies'] 
                                   if a.get('severity') == 'high']
                    print(f"   🚨 High Severity Anomalies: {len(high_severity)}")
                
                # Show column-specific anomalies
                if 'column_anomalies' in anomalies:
                    col_anomalies = anomalies['column_anomalies']
                    for col, stats in col_anomalies.items():
                        if stats['total_anomalies'] > 0:
                            print(f"   📊 {col}: {stats['total_anomalies']} anomalies")
            
        except Exception as e:
            print(f"   ❌ Anomaly detection failed: {str(e)}")


async def demonstrate_insight_generation(radar, datasets):
    """Demonstrate insight generation and ranking."""
    print("\n" + "="*60)
    print("💡 INSIGHT GENERATION & RANKING DEMONSTRATION")
    print("="*60)
    
    all_insights = []
    
    for name, data in datasets.items():
        print(f"\n🧠 Generating Insights for {name.replace('_', ' ').title()}...")
        
        try:
            results = await radar.detect_patterns(data)
            insights = results['insights']
            
            print(f"   💡 Insights Generated: {len(insights)}")
            print(f"   📊 Business Impact Score: {results['business_impact_score']:.3f}")
            
            # Categorize insights by priority
            high_priority = [i for i in insights if i['priority'] == 'high']
            medium_priority = [i for i in insights if i['priority'] == 'medium']
            low_priority = [i for i in insights if i['priority'] == 'low']
            
            print(f"   🔴 High Priority: {len(high_priority)}")
            print(f"   🟡 Medium Priority: {len(medium_priority)}")
            print(f"   🟢 Low Priority: {len(low_priority)}")
            
            # Show top insights
            for i, insight in enumerate(insights[:2]):
                print(f"   {i+1}. [{insight['priority'].upper()}] {insight['title']}")
                print(f"      {insight['description']}")
                print(f"      Impact: {insight['impact_score']:.3f} | Type: {insight['type']}")
            
            all_insights.extend(insights)
            
        except Exception as e:
            print(f"   ❌ Insight generation failed: {str(e)}")
    
    # Overall insight summary
    if all_insights:
        print(f"\n📋 OVERALL INSIGHT SUMMARY")
        print(f"   Total Insights: {len(all_insights)}")
        
        # Group by type
        insight_types = {}
        for insight in all_insights:
            insight_type = insight['type']
            if insight_type not in insight_types:
                insight_types[insight_type] = 0
            insight_types[insight_type] += 1
        
        for insight_type, count in insight_types.items():
            print(f"   {insight_type.title()}: {count}")
        
        # Top insights across all datasets
        all_insights.sort(key=lambda x: x['impact_score'], reverse=True)
        print(f"\n🏆 TOP 3 INSIGHTS ACROSS ALL DATASETS:")
        for i, insight in enumerate(all_insights[:3]):
            print(f"   {i+1}. {insight['title']} (Impact: {insight['impact_score']:.3f})")


async def demonstrate_statistical_testing(radar, datasets):
    """Demonstrate statistical significance testing."""
    print("\n" + "="*60)
    print("📊 STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)
    
    for name, data in datasets.items():
        print(f"\n🔬 Statistical Tests for {name.replace('_', ' ').title()}...")
        
        try:
            tests = await radar._perform_statistical_tests(data)
            
            # Correlation tests
            if 'correlation_tests' in tests:
                corr_tests = tests['correlation_tests']
                significant_corrs = [t for t in corr_tests if t['is_significant']]
                print(f"   🔗 Correlation Tests: {len(corr_tests)} total, {len(significant_corrs)} significant")
                
                for test in significant_corrs[:3]:  # Show top 3
                    print(f"      {test['variable_1']} ↔ {test['variable_2']}: "
                          f"r = {test['correlation']:.3f}, p = {test['p_value']:.4f}")
            
            # Normality tests
            if 'normality_tests' in tests:
                norm_tests = tests['normality_tests']
                normal_vars = [t for t in norm_tests if t['is_normal']]
                print(f"   📈 Normality Tests: {len(norm_tests)} variables, {len(normal_vars)} normally distributed")
            
        except Exception as e:
            print(f"   ❌ Statistical testing failed: {str(e)}")


async def demonstrate_notification_system(radar, datasets):
    """Demonstrate automated notification system."""
    print("\n" + "="*60)
    print("🔔 AUTOMATED NOTIFICATION SYSTEM")
    print("="*60)
    
    total_notifications = 0
    
    for name, data in datasets.items():
        print(f"\n📱 Checking Notifications for {name.replace('_', ' ').title()}...")
        
        try:
            results = await radar.detect_patterns(data)
            insights = results['insights']
            
            # Send notifications for high-priority insights
            notification_sent = await radar.send_insight_notification(insights, f"user_{name}")
            
            if notification_sent:
                high_priority_count = len([i for i in insights if i['priority'] == 'high'])
                print(f"   ✅ Notification sent for {high_priority_count} high-priority insights")
                total_notifications += 1
            else:
                print(f"   ℹ️  No high-priority insights requiring notification")
            
        except Exception as e:
            print(f"   ❌ Notification check failed: {str(e)}")
    
    print(f"\n📊 Total Notifications Sent: {total_notifications}")


async def demonstrate_health_monitoring(radar):
    """Demonstrate health monitoring capabilities."""
    print("\n" + "="*60)
    print("🏥 HEALTH MONITORING DEMONSTRATION")
    print("="*60)
    
    try:
        health = await radar.get_health_status()
        
        print(f"   Engine: {health['engine']}")
        print(f"   Version: {health['version']}")
        print(f"   Status: {health['status']}")
        print(f"   Last Check: {health['last_check']}")
        print(f"   Capabilities: {len(health['capabilities'])}")
        
        for capability in health['capabilities']:
            print(f"     ✓ {capability}")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {str(e)}")


def create_visualization_summary(datasets):
    """Create visualization summary of the datasets."""
    print("\n" + "="*60)
    print("📊 DATASET VISUALIZATION SUMMARY")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ScrollInsightRadar Demo Datasets Overview', fontsize=16)
        
        # Sales & Marketing Dataset
        if 'sales_marketing' in datasets:
            data = datasets['sales_marketing']
            axes[0, 0].plot(data['date'], data['sales'], label='Sales', alpha=0.7)
            axes[0, 0].plot(data['date'], data['marketing_spend'], label='Marketing Spend', alpha=0.7)
            axes[0, 0].set_title('Sales & Marketing Trends')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Financial Dataset
        if 'financial' in datasets:
            data = datasets['financial']
            axes[0, 1].plot(data['date'], data['stock_price'], color='green', alpha=0.7)
            axes[0, 1].set_title('Stock Price Movement')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # E-commerce Clustering
        if 'ecommerce' in datasets:
            data = datasets['ecommerce']
            scatter = axes[1, 0].scatter(data['avg_order_value'], data['orders_per_month'], 
                                       c=data['segment'].astype('category').cat.codes, 
                                       alpha=0.6, cmap='viridis')
            axes[1, 0].set_xlabel('Average Order Value')
            axes[1, 0].set_ylabel('Orders per Month')
            axes[1, 0].set_title('Customer Segmentation')
        
        # IoT Sensor Data
        if 'iot_sensors' in datasets:
            data = datasets['iot_sensors'].head(168)  # Show one week
            axes[1, 1].plot(data['timestamp'], data['temperature'], label='Temperature', alpha=0.7)
            axes[1, 1].plot(data['timestamp'], data['humidity'], label='Humidity', alpha=0.7)
            axes[1, 1].set_title('IoT Sensor Readings (1 Week)')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('scrollinsight_radar_demo_datasets.png', dpi=300, bbox_inches='tight')
        print("   📊 Visualization saved as 'scrollinsight_radar_demo_datasets.png'")
        
    except ImportError:
        print("   ℹ️  Matplotlib not available for visualization")
    except Exception as e:
        print(f"   ❌ Visualization failed: {str(e)}")


async def main():
    """Main demonstration function."""
    print("🚀 ScrollInsightRadar Comprehensive Demo")
    print("="*60)
    print("This demo showcases the advanced pattern detection,")
    print("trend analysis, and insight generation capabilities")
    print("of the ScrollInsightRadar engine.")
    print("="*60)
    
    # Initialize the radar engine
    print("\n🔧 Initializing ScrollInsightRadar Engine...")
    radar = ScrollInsightRadar()
    print(f"   ✅ Engine initialized: {radar.name} v{radar.version}")
    print(f"   🎯 Capabilities: {len(radar.capabilities)}")
    
    # Create sample datasets
    print("\n📊 Creating Sample Datasets...")
    datasets = create_sample_datasets()
    print(f"   ✅ Created {len(datasets)} datasets for analysis")
    
    # Run demonstrations
    await demonstrate_health_monitoring(radar)
    await demonstrate_pattern_detection(radar, datasets)
    await demonstrate_trend_analysis(radar, datasets)
    await demonstrate_anomaly_detection(radar, datasets)
    await demonstrate_insight_generation(radar, datasets)
    await demonstrate_statistical_testing(radar, datasets)
    await demonstrate_notification_system(radar, datasets)
    
    # Create visualizations
    create_visualization_summary(datasets)
    
    print("\n" + "="*60)
    print("✅ ScrollInsightRadar Demo Completed Successfully!")
    print("="*60)
    print("Key Features Demonstrated:")
    print("  🔍 Automated Pattern Detection")
    print("  📈 Statistical Trend Analysis")
    print("  ⚠️  Advanced Anomaly Detection")
    print("  💡 Intelligent Insight Generation")
    print("  📊 Statistical Significance Testing")
    print("  🔔 Automated Notification System")
    print("  🏥 Health Monitoring")
    print("\nThe ScrollInsightRadar engine is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())