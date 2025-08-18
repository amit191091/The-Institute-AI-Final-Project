"""
Diagnosis Submenu Module
Handles all diagnosis related menu options and functionality
"""

def display_diagnosis_menu(agent):
    """Display submenu for diagnosis"""
    while True:
        print("\n🔍 DIAGNOSIS SUBMENU")
        print("=" * 35)
        print("1. Quick Diagnosis (use saved files)")
        print("2. Complete Diagnosis (re-analyze all)")
        print("3. Force Re-analysis (overwrite saved files)")
        print("4. Regenerate Enhanced Results (with per_tooth_analysis)")
        print("5. Regenerate Enhanced Results")
        print("6. View Diagnosis Results")
        print("7. Generate Comprehensive Report")
        print("8. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                print("\n🚀 Running Quick Diagnosis...")
                results = agent.run_quick_diagnosis()
                if results:
                    agent.diagnosis_results = results
                    print("✅ Quick diagnosis completed successfully!")
                    print("\n💾 Auto-saving results...")
                    agent.save_results()
                    agent.save_picture_analysis_results()
                    agent.generate_comprehensive_report()
                    print("✅ Results automatically saved!")
                else:
                    print("❌ Quick diagnosis failed!")
                    
            elif choice == "2":
                print("\n🔄 Running Complete Diagnosis...")
                results = agent.run_complete_diagnosis()
                if results:
                    agent.diagnosis_results = results
                    print("✅ Complete diagnosis completed successfully!")
                    print("\n💾 Auto-saving results...")
                    agent.save_results()
                    agent.save_picture_analysis_results()
                    agent.generate_comprehensive_report()
                    print("✅ Results automatically saved!")
                else:
                    print("❌ Complete diagnosis failed!")
                    
            elif choice == "3":
                print("\n🔄 Running Force Re-analysis...")
                results = agent.run_complete_diagnosis()
                # Force re-analysis for all components
                agent.run_picture_analysis(use_saved_files=False, force_reanalysis=True)
                agent.run_vibration_analysis(use_saved_files=False, force_reanalysis=True)
                if results:
                    agent.diagnosis_results = results
                    print("✅ Force re-analysis completed successfully!")
                    print("\n💾 Auto-saving results...")
                    agent.save_results()
                    agent.save_picture_analysis_results()
                    agent.generate_comprehensive_report()
                    print("✅ Results automatically saved!")
                else:
                    print("❌ Force re-analysis failed!")
                    
            elif choice == "4":
                print("\n🔄 Regenerating Enhanced Results with per_tooth_analysis...")
                results = agent.regenerate_wear_measurement_results()
                if results:
                    print("✅ Enhanced results regenerated successfully!")
                    print("\n💾 Auto-saving enhanced results...")
                    agent.save_results()
                    agent.save_picture_analysis_results()
                    print("✅ Enhanced results automatically saved!")
                else:
                    print("❌ Enhanced results regeneration failed!")
                    
            elif choice == "5":
                print("\n🔄 Regenerating Enhanced Results with per_tooth_analysis...")
                results = agent.regenerate_wear_measurement_results()
                if results:
                    print("✅ Enhanced results regenerated successfully!")
                    print("\n💾 Auto-saving enhanced results...")
                    agent.save_results()
                    agent.save_picture_analysis_results()
                    print("✅ Enhanced results automatically saved!")
                else:
                    print("❌ Enhanced results regeneration failed!")
                    
            elif choice == "6":
                if hasattr(agent, 'diagnosis_results') and agent.diagnosis_results:
                    display_diagnosis_results_menu(agent, agent.diagnosis_results)
                else:
                    print("⚠️ No diagnosis results available. Run diagnosis first.")
                    
            elif choice == "7":
                print("\n📝 Generating Comprehensive Report...")
                agent.generate_comprehensive_report()
                print("✅ Comprehensive report generated!")
                
            elif choice == "8":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_diagnosis_results_menu(agent, results):
    """Display menu for diagnosis results"""
    while True:
        print("\n📊 DIAGNOSIS RESULTS")
        print("=" * 35)
        print("1. View Overall Diagnosis")
        print("2. View Picture Analysis Results")
        print("3. View Vibration Analysis Results")
        print("4. View Correlation Analysis")
        print("5. Save Diagnosis Results")
        print("6. Return to Diagnosis Menu")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🔍 OVERALL DIAGNOSIS:")
                print("=" * 30)
                if 'overall_assessment' in results:
                    assessment = results['overall_assessment']
                    print(f"   Overall Status: {assessment.get('status', 'Unknown')}")
                    print(f"   Confidence Level: {assessment.get('confidence', 'Unknown')}")
                    print(f"   Recommendations: {assessment.get('recommendations', 'None')}")
                else:
                    print("   No overall assessment available")
                    
            elif choice == "2":
                if 'picture_analysis' in results:
                    print("\n🖼️ PICTURE ANALYSIS RESULTS:")
                    print("=" * 35)
                    pic_results = results['picture_analysis']
                    for case, data in pic_results.items():
                        print(f"\n🔧 {case}:")
                        if 'wear_depth' in data:
                            print(f"   Wear Depth: {data['wear_depth']:.2f} μm")
                        if 'wear_area' in data:
                            print(f"   Wear Area: {data['wear_area']:.2f} mm²")
                else:
                    print("⚠️ No picture analysis results available")
                    
            elif choice == "3":
                if 'vibration_analysis' in results:
                    print("\n📊 VIBRATION ANALYSIS RESULTS:")
                    print("=" * 35)
                    vib_results = results['vibration_analysis']
                    for case, data in vib_results.items():
                        print(f"\n📊 {case}:")
                        if 'rms' in data:
                            print(f"   RMS: {data['rms']:.4f}")
                        if 'dominant_frequency' in data:
                            print(f"   Dominant Frequency: {data['dominant_frequency']:.2f} Hz")
                else:
                    print("⚠️ No vibration analysis results available")
                    
            elif choice == "4":
                print("\n📈 Generating Correlation Analysis...")
                agent.plot_correlation_analysis()
                print("✅ Correlation analysis generated!")
                    
            elif choice == "5":
                print("\n💾 Saving Diagnosis Results...")
                agent.save_results()
                print("✅ Diagnosis results saved successfully!")
                
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
