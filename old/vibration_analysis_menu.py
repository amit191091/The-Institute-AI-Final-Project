"""
Vibration Analysis Submenu Module
Handles all vibration analysis related menu options and functionality
"""

def display_vibration_analysis_menu(agent):
    """Display submenu for vibration analysis"""
    while True:
        print("\n📊 VIBRATION ANALYSIS SUBMENU")
        print("=" * 35)
        print("1. Database Analysis")
        print("2. View Vibration Analysis Report")
        print("3. View Wear Level Summary")
        print("4. View Overall Assessment")
        print("5. Return to Main Menu")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n🔄 Running Vibration Analysis with Force Re-analysis...")
                results = agent.run_vibration_analysis(use_saved_files=False, force_reanalysis=True)
                if results:
                    agent.vibration_analysis_results = results
                    print("✅ Vibration analysis completed successfully!")
                else:
                    print("❌ Vibration analysis failed!")
                    
            elif choice == "2":
                if hasattr(agent, 'vibration_analysis_results') and agent.vibration_analysis_results:
                    display_vibration_results_menu(agent, agent.vibration_analysis_results)
                else:
                    print("⚠️ No vibration analysis results available. Run analysis first.")
                    
            elif choice == "3":
                if hasattr(agent, 'vibration_analysis_results') and agent.vibration_analysis_results:
                    display_vibration_wear_level_summary(agent.vibration_analysis_results)
                else:
                    print("⚠️ No vibration analysis results available. Run analysis first.")
                    
            elif choice == "4":
                if hasattr(agent, 'vibration_analysis_results') and agent.vibration_analysis_results:
                    display_vibration_overall_assessment(agent.vibration_analysis_results)
                else:
                    print("⚠️ No vibration analysis results available. Run analysis first.")
                    
            elif choice == "5":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-5.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_vibration_results_menu(agent, results):
    """Display menu for vibration analysis results"""
    while True:
        print("\n📊 VIBRATION ANALYSIS RESULTS")
        print("=" * 35)
        print("1. View Detailed Results")
        print("2. View Frequency Response")
        print("3. View Time Domain Analysis")
        print("4. View FFT Analysis")
        print("5. Save Results")
        print("6. Return to Vibration Analysis Menu")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n📋 DETAILED VIBRATION RESULTS:")
                print("=" * 35)
                for case, data in results.items():
                    print(f"\n📊 {case}:")
                    if 'rms' in data:
                        print(f"   RMS Value: {data['rms']:.4f}")
                    if 'peak' in data:
                        print(f"   Peak Value: {data['peak']:.4f}")
                    if 'crest_factor' in data:
                        print(f"   Crest Factor: {data['crest_factor']:.4f}")
                    if 'kurtosis' in data:
                        print(f"   Kurtosis: {data['kurtosis']:.4f}")
                    if 'dominant_frequency' in data:
                        print(f"   Dominant Frequency: {data['dominant_frequency']:.2f} Hz")
                        
            elif choice == "2":
                print("\n📈 Generating Frequency Response...")
                agent.vibration_analyzer.plot_frequency_response()
                print("✅ Frequency response plot generated!")
                
            elif choice == "3":
                print("\n⏰ Generating Time Domain Analysis...")
                agent.vibration_analyzer.plot_time_domain_analysis()
                print("✅ Time domain analysis plot generated!")
                
            elif choice == "4":
                print("\n📊 Generating FFT Analysis...")
                agent.vibration_analyzer.plot_fft_analysis()
                print("✅ FFT analysis plot generated!")
                
            elif choice == "5":
                print("\n💾 Saving Results...")
                agent.vibration_analyzer.save_results()
                print("✅ Results saved successfully!")
                
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_vibration_wear_level_summary(results):
    """Display vibration wear level summary"""
    print("\n📊 VIBRATION WEAR LEVEL SUMMARY")
    print("=" * 35)
    
    # Group by wear level
    wear_levels = {}
    for case, data in results.items():
        if 'rms' in data:
            wear_levels[case] = data['rms']
    
    if wear_levels:
        # Sort by RMS value (vibration level)
        sorted_levels = sorted(wear_levels.items(), key=lambda x: x[1])
        
        print("\n📊 Vibration Levels (from lowest to highest):")
        for case, rms_value in sorted_levels:
            print(f"   {case}: {rms_value:.4f} RMS")
            
        # Calculate statistics
        rms_values = list(wear_levels.values())
        print(f"\n📈 Statistics:")
        print(f"   Min RMS: {min(rms_values):.4f}")
        print(f"   Max RMS: {max(rms_values):.4f}")
        print(f"   Average RMS: {sum(rms_values)/len(rms_values):.4f}")
    else:
        print("⚠️ No vibration level data available")

def display_vibration_overall_assessment(results):
    """Display overall assessment of vibration analysis"""
    print("\n🔍 VIBRATION OVERALL ASSESSMENT")
    print("=" * 35)
    
    total_cases = len(results)
    cases_with_vibration = sum(1 for data in results.values() if 'rms' in data and data['rms'] > 0)
    
    print(f"\n📊 Summary:")
    print(f"   Total Cases Analyzed: {total_cases}")
    print(f"   Cases with Vibration Data: {cases_with_vibration}")
    
    if cases_with_vibration > 0:
        rms_values = [data['rms'] for data in results.values() if 'rms' in data and data['rms'] > 0]
        avg_rms = sum(rms_values) / len(rms_values)
        
        print(f"\n📊 Vibration Analysis:")
        print(f"   Average RMS: {avg_rms:.4f}")
        
        if avg_rms < 0.1:
            print("   Assessment: Low vibration levels detected")
        elif avg_rms < 0.5:
            print("   Assessment: Moderate vibration levels detected")
        else:
            print("   Assessment: High vibration levels detected")
    
    print(f"\n✅ Vibration analysis assessment complete!")
