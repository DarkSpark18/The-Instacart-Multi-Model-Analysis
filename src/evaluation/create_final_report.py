# create_final_report.py
"""
Master script to generate comprehensive analysis and final report

Run this single file to create all visualizations and reports!
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("\n" + "="*70)
    print("🚀 GENERATING COMPREHENSIVE FINAL REPORT")
    print("="*70 + "\n")
    
    # Step 1: Run analysis
    print("Step 1/2: Running comprehensive analysis...")
    print("-" * 70)
    try:
        result = subprocess.run([sys.executable, "analyze_results.py"], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ Analysis failed!")
            return
    except FileNotFoundError:
        print("❌ analyze_results.py not found!")
        print("Make sure the file is in the current directory.")
        return
    
    # Step 2: Generate HTML report
    print("\n\nStep 2/2: Generating HTML report...")
    print("-" * 70)
    try:
        result = subprocess.run([sys.executable, "generate_report.py"],
                              capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ Report generation failed!")
            return
    except FileNotFoundError:
        print("❌ generate_report.py not found!")
        return
    
    print("\n" + "="*70)
    print("✅ ALL REPORTS GENERATED SUCCESSFULLY!")
    print("="*70)
    
    # Summary
    print("\n📁 OUTPUT FILES:")
    print("-" * 70)
    
    analysis_dir = Path("outputs/analysis")
    if analysis_dir.exists():
        print("\n📊 Visualizations:")
        for img in analysis_dir.glob("*.png"):
            print(f"   • {img.name}")
        
        print("\n📄 CSV Reports:")
        for csv in analysis_dir.glob("*.csv"):
            print(f"   • {csv.name}")
    
    report_path = Path("outputs/FINAL_REPORT.html")
    if report_path.exists():
        print(f"\n🌐 Interactive Report:")
        print(f"   • {report_path.name}")
        print(f"\n   ➜ Open this file in your browser!")
        print(f"   ➜ Location: {report_path.absolute()}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()