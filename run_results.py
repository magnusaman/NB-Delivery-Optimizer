#!/usr/bin/env python3
"""
Simple script to run the results analysis
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

# Import and run the analyzer
try:
    from results import DeliveryResultsAnalyzer
    
    print("📊 DELIVERY SYSTEM RESULTS ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DeliveryResultsAnalyzer()
    
    # Create comprehensive dashboard
    analyzer.create_comprehensive_dashboard()
    
    # Generate detailed report
    analyzer.generate_detailed_report()
    
    print("\n✅ RESULTS ANALYSIS COMPLETED!")
    print("📁 Generated files:")
    print("   📊 delivery_dashboard.png - Comprehensive dashboard")
    print("   📋 delivery_report.txt - Detailed text report")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
