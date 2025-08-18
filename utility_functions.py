"""
Utility Functions Module
Contains various utility functions for the gear wear diagnosis system
"""

import os
import json

def display_results_menu(agent, results):
    """Display menu for overall results"""
    while True:
        print("\n📊 OVERALL RESULTS")
        print("=" * 35)
        print("1. View Overall Assessment")
        print("2. View Picture Analysis Results")
        print("3. View Vibration Analysis Results")
        print("4. View Wear Case Details")
        print("5. Save Results")
        print("6. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                if 'overall_assessment' in results:
                    assessment = results['overall_assessment']
                    print(f"\n🔍 Overall Status: {assessment.get('status', 'Unknown')}")
                    print(f"📊 Confidence Level: {assessment.get('confidence', 'Unknown')}")
                    print(f"💡 Recommendations: {assessment.get('recommendations', 'None')}")
                else:
                    print("⚠️ No overall assessment available")
                    
            elif choice == "2":
                if 'picture_analysis' in results and results['picture_analysis']:
                    display_picture_summary(results['picture_analysis'])
                else:
                    print("⚠️ No picture analysis results available")
                    
            elif choice == "3":
                if 'vibration_analysis' in results and results['vibration_analysis']:
                    display_vibration_summary(results['vibration_analysis'])
                else:
                    print("⚠️ No vibration analysis results available")
                    
            elif choice == "4":
                display_wear_case_details(results)
                    
            elif choice == "5":
                print("\n💾 Saving Results...")
                agent.save_results()
                print("✅ Results saved successfully!")
                
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_wear_case_details(results, wear_level):
    """Display detailed information for a specific wear case"""
    if not results or 'wear_level_analysis' not in results:
        print("❌ No wear analysis data available.")
        return
    
    wear_levels = results['wear_level_analysis']
    if wear_level not in wear_levels:
        print(f"❌ Wear level {wear_level} not found in analysis results.")
        return
    
    analysis = wear_levels[wear_level]
    print(f"\n🔍 WEAR LEVEL {wear_level} DETAILED ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print(f"📊 Images Analyzed: {analysis['image_count']}")
    print(f"📈 Wear Progression Score: {analysis['wear_progression_score']:.3f}")
    print(f"📏 Average Addendum Height: {analysis['avg_addendum_height']:.3f}")
    print(f"📏 Average Dedendum Height: {analysis['avg_dedendum_height']:.3f}")
    print(f"📏 Average Tooth Thickness: {analysis['avg_tooth_thickness']:.3f}")
    print(f"📐 Average Tooth Area: {analysis['avg_tooth_area']:.3f}")
    
    # Deviation from healthy
    if 'deviation_from_healthy' in analysis:
        dev = analysis['deviation_from_healthy']
        print(f"🔄 Overall Deviation Score: {dev.get('overall_deviation_score', 0):.3f}")
    
    # Individual tooth analysis
    if 'individual_teeth' in analysis and analysis['individual_teeth']:
        print(f"\n⚙️ INDIVIDUAL TOOTH ANALYSIS")
        print("-" * 40)
        
        # Sort teeth numerically
        sorted_teeth = sorted(analysis['individual_teeth'].keys(), key=lambda x: int(x))
        analyzed_count = 0
        missing_count = 0
        
        for tooth_num in sorted_teeth:
            tooth_data = analysis['individual_teeth'][tooth_num]
            if tooth_data.get('status') == 'missing':
                missing_count += 1
                print(f"   Tooth {tooth_num}: ❌ Missing")
            else:
                analyzed_count += 1
                if 'comparison' in tooth_data and tooth_data['comparison']:
                    comp = tooth_data['comparison']
                    print(f"   Tooth {tooth_num}: Score {comp['overall_wear_score']:.3f}, "
                          f"Thickness Wear: {comp.get('thickness_wear_depth', 'N/A')} (Primary), "
                          f"Addendum Wear: {comp.get('addendum_wear_depth', 'N/A')} (Calibration), "
                          f"Damage: {comp['damage_severity']}")
                else:
                    print(f"   Tooth {tooth_num}: ✅ Analyzed (no comparison data)")
        
        print(f"\n📊 Summary: {analyzed_count} teeth analyzed, {missing_count} teeth missing")

def display_system_information():
    """Display system information"""
    print("\nℹ️ SYSTEM INFORMATION")
    print("=" * 35)
    print("🔧 Gear Wear Diagnosis System")
    print("📊 Version: 1.0")
    print("🔍 Analysis Types:")
    print("   • Picture Analysis (Visual wear detection)")
    print("   • Vibration Analysis (Mechanical response)")
    print("\n📁 Data Locations:")
    print("   • Picture Data: gear_images/database/")
    print("   • Vibration Data: vibration_data/database/")
    print("\n💾 Output Files:")
    print("   • comprehensive_diagnosis_report.txt")
    print("   • comprehensive_diagnosis_results.json")
    print("   • gear_diagnosis_report.txt")
    
    # Display saved files status
    check_saved_files_status()

def display_picture_summary(picture_results):
    """Display picture analysis summary"""
    print("\n🖼️ PICTURE ANALYSIS SUMMARY")
    print("=" * 35)
    
    if picture_results:
        total_cases = len(picture_results)
        cases_with_wear = sum(1 for data in picture_results.values() if 'wear_depth' in data and data['wear_depth'] > 0)
        
        print(f"📊 Total Cases: {total_cases}")
        print(f"🔧 Cases with Wear: {cases_with_wear}")
        print(f"✅ Healthy Cases: {total_cases - cases_with_wear}")
        
        if cases_with_wear > 0:
            wear_depths = [data['wear_depth'] for data in picture_results.values() if 'wear_depth' in data and data['wear_depth'] > 0]
            avg_wear = sum(wear_depths) / len(wear_depths)
            print(f"📈 Average Wear Depth: {avg_wear:.2f} μm")
    else:
        print("⚠️ No picture analysis results available")


def display_vibration_summary(vibration_results):
    """Display vibration analysis summary"""
    print("\n📊 VIBRATION ANALYSIS SUMMARY")
    print("=" * 35)
    
    if vibration_results:
        total_cases = len(vibration_results)
        cases_with_vibration = sum(1 for data in vibration_results.values() if 'rms' in data and data['rms'] > 0)
        
        print(f"📊 Total Cases: {total_cases}")
        print(f"📊 Cases with Vibration Data: {cases_with_vibration}")
        
        if cases_with_vibration > 0:
            rms_values = [data['rms'] for data in vibration_results.values() if 'rms' in data and data['rms'] > 0]
            avg_rms = sum(rms_values) / len(rms_values)
            print(f"📈 Average RMS: {avg_rms:.4f}")
    else:
        print("⚠️ No vibration analysis results available")

def check_saved_files_status():
    """Check and display status of saved files"""
    print("\n📁 SAVED FILES STATUS")
    print("=" * 25)
    
    import os
    
    files_to_check = [
        ("gear_images/healthy_baseline.json", "Healthy Baseline Analysis"),
        ("gear_images/wear_analysis_results.json", "Wear Analysis Results"),
        ("comprehensive_diagnosis_results.json", "Comprehensive Results"),
        ("comprehensive_diagnosis_report.txt", "Comprehensive Report"),
        ("gear_images/picture_analysis_report.txt", "Picture Analysis Report"),
        ("gear_images/picture_analysis_results.json", "Picture Analysis Results")
    ]
    
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {description}")
            print(f"   📄 {filepath}")
            print(f"   📏 Size: {size:,} bytes")
        else:
            print(f"❌ {description}")
            print(f"   📄 {filepath} - Not found")
        print()
