"""
Write Summary Submenu Module
Handles all summary writing related menu options and functionality
"""

def display_write_summary_menu(agent):
    """Display submenu for write summary"""
    while True:
        print("\n📝 WRITE SUMMARY SUBMENU")
        print("=" * 35)
        print("1. Generate Picture Analysis Summary")
        print("2. Generate Vibration Analysis Summary")
        print("3. Generate Comprehensive Summary")
        print("4. Generate Technical Report")
        print("5. View Saved Reports")
        print("6. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n📝 Generating Picture Analysis Summary...")
                if hasattr(agent, 'picture_analysis_results') and agent.picture_analysis_results:
                    agent.generate_picture_summary()
                    print("✅ Picture analysis summary generated!")
                else:
                    print("⚠️ No picture analysis results available. Run analysis first.")
                    
            elif choice == "2":
                print("\n📝 Vibration Analysis")
                print("ℹ️ Use the Vibration Analysis submenu to view plots")
                print("ℹ️ No data processing or report generation is performed")
                    
            elif choice == "3":
                print("\n📝 Generating Comprehensive Summary...")
                agent.generate_comprehensive_report()
                print("✅ Comprehensive summary generated!")
                
            elif choice == "4":
                print("\n📝 Generating Technical Report...")
                agent.generate_technical_report()
                print("✅ Technical report generated!")
                
            elif choice == "5":
                display_saved_reports_menu()
                
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_saved_reports_menu():
    """Display menu for viewing saved reports"""
    while True:
        print("\n📁 SAVED REPORTS")
        print("=" * 35)
        print("1. View Picture Analysis Report")
        print("2. View Vibration Analysis Report")
        print("3. View Comprehensive Report")
        print("4. View Technical Report")
        print("5. List All Report Files")
        print("6. Return to Write Summary Menu")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n📄 PICTURE ANALYSIS REPORT:")
                print("=" * 35)
                try:
                    with open('picture_analysis_report.txt', 'r') as f:
                        content = f.read()
                        print(content)
                except FileNotFoundError:
                    print("⚠️ Picture analysis report not found")
                    
            elif choice == "2":
                print("\n📄 VIBRATION ANALYSIS REPORT:")
                print("=" * 35)
                print("ℹ️ Vibration analysis")
                print("ℹ️ No reports are generated")
                    
            elif choice == "3":
                print("\n📄 COMPREHENSIVE REPORT:")
                print("=" * 35)
                try:
                    with open('comprehensive_diagnosis_report.txt', 'r') as f:
                        content = f.read()
                        print(content)
                except FileNotFoundError:
                    print("⚠️ Comprehensive report not found")
                    
            elif choice == "4":
                print("\n📄 TECHNICAL REPORT:")
                print("=" * 35)
                try:
                    with open('technical_report.txt', 'r') as f:
                        content = f.read()
                        print(content)
                except FileNotFoundError:
                    print("⚠️ Technical report not found")
                    
            elif choice == "5":
                print("\n📁 ALL REPORT FILES:")
                print("=" * 35)
                import os
                report_files = [
                    'picture_analysis_report.txt',
                    'vibration_analysis_report.txt',
                    'comprehensive_diagnosis_report.txt',
                    'technical_report.txt',
                    'gear_diagnosis_report.txt'
                ]
                
                for file in report_files:
                    if os.path.exists(file):
                        file_size = os.path.getsize(file)
                        print(f"   ✅ {file} ({file_size} bytes)")
                    else:
                        print(f"   ❌ {file} (not found)")
                        
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
