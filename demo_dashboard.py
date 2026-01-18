#!/usr/bin/env python3
"""
Clinical Dashboard Demo

Demonstrates the real-time clinical dashboard for depression detection
monitoring and management.

Usage:
    streamlit run demo_dashboard.py

Features:
- Real-time risk monitoring
- Multi-patient session management
- Alert management and acknowledgment
- Clinical workflow integration
- Role-based access control

Demo Credentials:
- Admin: admin / admin123
- Clinician: dr_smith / clinic123
- Observer: observer / observe123
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import and run dashboard
from dashboard.main_dashboard import main

if __name__ == "__main__":
    print("🏥 Starting Clinical Dashboard...")
    print("📊 Real-time Depression Detection Monitoring")
    print("🔐 Demo Credentials:")
    print("   Admin: admin / admin123")
    print("   Clinician: dr_smith / clinic123") 
    print("   Observer: observer / observe123")
    print("🌐 Dashboard will open in your browser...")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)