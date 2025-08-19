import os
import sys
import json
import numpy as np
from datetime import datetime

# Add gear_images to path for imports
sys.path.append('gear_images')

# Import analysis modules
from vibration_analysis import VibrationAnalysis

# Import submenu modules
from picture_analysis_menu import display_picture_analysis_menu
from vibration_analysis_menu import display_vibration_analysis_menu
from diagnosis_menu import display_diagnosis_menu
from write_summary_menu import display_write_summary_menu

# Import utility functions
from utility_functions import (
    display_results_menu, 
    display_wear_case_details, 
    display_system_information,
    display_picture_summary,
    display_vibration_summary,
    check_saved_files_status
)

class GearWearDiagnosisAgent:
    """Main coordinator for gear wear diagnosis using modular analysis"""
    
    def __init__(self):
        self.vibration_analyzer = VibrationAnalysis()
        self.overall_diagnosis = {}
    
    def run_picture_analysis(self, use_saved_files=True, force_reanalysis=False):
        """Run picture analysis pipeline using integrated system"""
        print("🖼️ Starting Picture Analysis...")
        print("=" * 50)
        print("ℹ️ Picture analysis is now handled through the integrated menu system.")
        print("ℹ️ Please use 'Database Process (Gear Wear Analysis)' from the Picture Analysis menu.")
        print("ℹ️ This will run the complete analysis using the new integrated scripts.")
        
        # Return a placeholder result indicating the new system should be used
        return {
            "status": "integrated_system",
            "message": "Picture analysis is now handled through the integrated menu system",
            "files": {
                "single_tooth_results": "gear_images/single_tooth_results.csv",
                "all_teeth_results": "gear_images/all_teeth_results.csv",
            }
        }
    
    def run_quick_diagnosis(self):
        """Run quick diagnosis using saved files (fast mode)"""
        print("⚡ Quick Diagnosis Mode - Using Saved Files")
        print("=" * 60)
        
        # Initialize results dictionary
        self.overall_diagnosis = {
            'picture_analysis': None,
            'vibration_analysis': None,
            'overall_assessment': None,
            'recommendations': []
        }
        
        # Step 1: Quick Picture Analysis (use saved files)
        picture_results = self.run_picture_analysis(use_saved_files=True)
        self.overall_diagnosis['picture_analysis'] = picture_results
        
        
        # Step 2: Vibration Analysis (if data exists)
        vibration_results = self.run_vibration_analysis()
        self.overall_diagnosis['vibration_analysis'] = vibration_results
        
        # Step 3: Generate overall assessment
        self.overall_diagnosis['overall_assessment'] = self._calculate_overall_assessment()
        self.overall_diagnosis['recommendations'] = self._generate_recommendations()
        
        print("\n✅ Quick diagnosis finished!")
        return self.overall_diagnosis
    
    
    def run_vibration_analysis(self, vibration_paths=None):
        """Run vibration analysis"""
        print("📊 Starting Vibration Analysis...")
        print("=" * 50)
        
        if vibration_paths is None:
            # Default vibration file paths
            vibration_paths = [
                "vibration_data/vibration1.mat",
                "vibration_data/vibration2.mat"
            ]
        
        # Filter existing files
        existing_vibration = [path for path in vibration_paths if os.path.exists(path)]
        
        if existing_vibration:
            results = self.vibration_analyzer.analyze_vibration_data(existing_vibration)
            print("✅ Vibration analysis completed successfully")
            return results
        else:
            print("⚠️ No vibration data files found")
            return None
    
    def run_complete_diagnosis(self):
        """Run complete gear wear diagnosis with all analysis types"""
        print("🔧 Gear Wear Diagnosis Agent")
        print("=" * 60)
        
        # Initialize results dictionary
        self.overall_diagnosis = {
            'picture_analysis': None,
            'vibration_analysis': None,
            'overall_assessment': None,
            'recommendations': []
        }
        
        # Step 1: Picture Analysis
        picture_results = self.run_picture_analysis()
        self.overall_diagnosis['picture_analysis'] = picture_results
        
        
        # Step 2: Vibration Analysis
        vibration_results = self.run_vibration_analysis()
        self.overall_diagnosis['vibration_analysis'] = vibration_results
        
        # Step 3: Generate overall assessment
        self.overall_diagnosis['overall_assessment'] = self._calculate_overall_assessment()
        self.overall_diagnosis['recommendations'] = self._generate_recommendations()
        
        print("\n✅ Complete diagnosis finished!")
        return self.overall_diagnosis
    
    def _calculate_overall_assessment(self):
        """Calculate overall gear condition assessment"""
        scores = []
        
        # Picture analysis score
        if self.overall_diagnosis['picture_analysis']:
            picture_score = self.overall_diagnosis['picture_analysis']['overall_assessment']['overall_score']
            scores.append(('Picture', picture_score, 0.6))  # 60% weight
        
        
        # Vibration analysis score
        if self.overall_diagnosis['vibration_analysis']:
            vibration_scores = []
            for analysis in self.overall_diagnosis['vibration_analysis'].values():
                bearing_fault = analysis['bearing_analysis']['bearing_fault_probability']
                fault_score = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}[bearing_fault]
                vibration_scores.append(fault_score)
            vibration_score = np.mean(vibration_scores) if vibration_scores else 0
            scores.append(('Vibration', vibration_score, 0.2))  # 20% weight
        
        # Calculate weighted overall score
        if scores:
            overall_score = sum(score * weight for _, score, weight in scores)
            
            # Determine condition level
            if overall_score < 0.3:
                condition = "Good"
                status = "✅ Normal operation"
            elif overall_score < 0.6:
                condition = "Fair"
                status = "⚠️ Moderate wear detected"
            else:
                condition = "Poor"
                status = "🚨 Significant wear/damage detected"
            
            return {
                'overall_score': overall_score,
                'condition': condition,
                'status': status,
                'component_scores': {name: score for name, score, _ in scores}
            }
        else:
            return {
                'overall_score': 0,
                'condition': "Unknown",
                'status': "❌ No analysis data available",
                'component_scores': {}
            }
    
    def _generate_recommendations(self):
        """Generate maintenance recommendations based on analysis"""
        assessment = self.overall_diagnosis['overall_assessment']
        recommendations = []
        
        if assessment['overall_score'] < 0.3:
            recommendations.append("Continue routine maintenance schedule")
            recommendations.append("Monitor for any changes in operation")
        elif assessment['overall_score'] < 0.6:
            recommendations.append("Schedule maintenance within 1-2 months")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider lubricant analysis")
        else:
            recommendations.append("Immediate maintenance required")
            recommendations.append("Consider gear replacement")
            recommendations.append("Investigate root cause of wear")
            recommendations.append("Implement preventive measures")
        
        return recommendations
    
    def generate_comprehensive_report(self, output_path="comprehensive_diagnosis_report.txt"):
        """Generate a comprehensive diagnosis report"""
        if not self.overall_diagnosis:
            print("❌ No diagnosis available. Run analysis first.")
            return None
        
        report = []
        report.append("=" * 80)
        report.append("                    COMPREHENSIVE GEAR WEAR DIAGNOSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall Assessment
        assessment = self.overall_diagnosis['overall_assessment']
        report.append("📊 OVERALL ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Condition: {assessment['condition']}")
        report.append(f"Status: {assessment['status']}")
        report.append(f"Overall Score: {assessment['overall_score']:.3f}")
        report.append("")
        
        # Component Scores
        if 'component_scores' in assessment:
            report.append("📈 COMPONENT SCORES")
            report.append("-" * 40)
            for component, score in assessment['component_scores'].items():
                report.append(f"{component} Analysis: {score:.3f}")
            report.append("")
        
        # Picture Analysis Summary
        if self.overall_diagnosis['picture_analysis']:
            report.append("🖼️ PICTURE ANALYSIS SUMMARY")
            report.append("-" * 40)
            picture_assessment = self.overall_diagnosis['picture_analysis']['overall_assessment']
            report.append(f"Picture Analysis Score: {picture_assessment['overall_score']:.3f}")
            report.append(f"Picture Analysis Status: {picture_assessment['status']}")
            report.append("")
        
        
        # Vibration Analysis Summary
        if self.overall_diagnosis['vibration_analysis']:
            report.append("📊 VIBRATION ANALYSIS SUMMARY")
            report.append("-" * 40)
            for path, analysis in self.overall_diagnosis['vibration_analysis'].items():
                report.append(f"File: {analysis['filename']}")
                report.append(f"  - Bearing Fault Probability: {analysis['bearing_analysis']['bearing_fault_probability']}")
                if analysis['gear_mesh_analysis']['dominant_mesh_frequency']:
                    report.append(f"  - Dominant Mesh Frequency: {analysis['gear_mesh_analysis']['dominant_mesh_frequency']:.1f} Hz")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(self.overall_diagnosis['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        report.append("=" * 80)
        report.append("Report generated by Comprehensive Gear Wear Diagnosis Agent")
        report.append("=" * 80)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Comprehensive report saved to: {output_path}")
        return '\n'.join(report)
    
    def save_results(self, output_path="comprehensive_diagnosis_results.json"):
        """Save all analysis results to JSON file"""
        if not self.overall_diagnosis:
            print("❌ No results available. Run analysis first.")
            return False
        
        # Ensure per_tooth_analysis is included from wear measurement results
        if self.overall_diagnosis.get('picture_analysis') and isinstance(self.overall_diagnosis['picture_analysis'], dict):
            if 'per_tooth_analysis' in self.overall_diagnosis['picture_analysis']:
                # Already included
                pass
            else:
                # Try to load from wear_measurement_results.json if available
                try:
                    wear_results_path = "results/wear_measurement_results.json"
                    if os.path.exists(wear_results_path):
                        with open(wear_results_path, 'r') as f:
                            wear_results = json.load(f)
                        if 'per_tooth_analysis' in wear_results:
                            self.overall_diagnosis['picture_analysis']['per_tooth_analysis'] = wear_results['per_tooth_analysis']
                            print("✅ Added per_tooth_analysis data to comprehensive results")
                except Exception as e:
                    print(f"⚠️ Could not load per_tooth_analysis: {str(e)}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.overall_diagnosis, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_path}")
        return True
    
    def save_picture_analysis_results(self, output_path="results/picture_analysis_results.json"):
        """Save picture analysis results with per_image data"""
        if not self.overall_diagnosis or not self.overall_diagnosis.get('picture_analysis'):
            print("❌ No picture analysis results available. Run analysis first.")
            return False
        
        # Generate per_image data
        per_image_data = self._generate_per_image_data()
        
        # Create picture analysis results with per_image data
        picture_results = {
            'wear_level_analysis': self.overall_diagnosis['picture_analysis'].get('wear_level_analysis', {}),
            'healthy_reference': self.overall_diagnosis['picture_analysis'].get('healthy_reference', {}),
            'overall_assessment': self.overall_diagnosis['picture_analysis'].get('overall_assessment', {}),
            'recommendations': self.overall_diagnosis['picture_analysis'].get('recommendations', []),
            'per_image': per_image_data
        }
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(picture_results, f, indent=2, default=str)
        
        print(f"✅ Picture analysis results saved to: {output_path}")
        return True
    
    def _generate_per_image_data(self):
        """Generate per_image data for picture analysis results"""
        per_image_data = []
        
        try:
            # Try to load wear measurement results to get image data
            wear_results_path = "results/wear_measurement_results.json"
            if os.path.exists(wear_results_path):
                with open(wear_results_path, 'r') as f:
                    wear_results = json.load(f)
                
                # Extract image data from wear analysis
                if 'wear_analysis' in wear_results:
                    for wear_case, case_data in wear_results['wear_analysis'].items():
                        wear_level = case_data.get('wear_level', 'Unknown')
                        if 'tooth_analysis' in case_data:
                            for tooth_id, tooth_data in case_data['tooth_analysis'].items():
                                # Extract orientation and scale information
                                alignment_info = tooth_data.get('alignment_info', {})
                                scale_factor = tooth_data.get('scale_factor', 250.0)
                                
                                image_record = {
                                    'image_path': tooth_data.get('image_path', ''),
                                    'wear_case': wear_case,
                                    'tooth_id': int(tooth_id),
                                    'scale_factor': scale_factor,
                                    'pixels_per_mm': scale_factor / 1000.0,  # Convert to pixels/mm
                                    'orientation_info': {
                                        'aligned': alignment_info.get('aligned', False),
                                        'original_orientation': alignment_info.get('original_orientation', 0.0),
                                        'aligned_orientation': alignment_info.get('aligned_orientation', 0.0),
                                        'rotation_applied': alignment_info.get('aligned_orientation', 0.0) - alignment_info.get('original_orientation', 0.0)
                                    },
                                    'case_status': f"Wear Level {wear_level}",
                                    'tooth_status': f"Tooth {tooth_id}",
                                    'wear_measurements': {
                                        'area_wear_percent': tooth_data.get('area_wear', {}).get('percentage', 0.0),
                                        'perimeter_wear_percent': tooth_data.get('perimeter_wear', {}).get('percentage', 0.0),
                                        'wear_score': tooth_data.get('wear_score', 0.0),
                                        'area_loss': tooth_data.get('area_wear', {}).get('area_loss', 0.0),
                                        'perimeter_loss': tooth_data.get('perimeter_wear', {}).get('perimeter_loss', 0.0)
                                    }
                                }
                                per_image_data.append(image_record)
                
                # Also add healthy baseline images
                if 'healthy_baseline' in wear_results:
                    for tooth_id, tooth_data in wear_results['healthy_baseline'].items():
                        orientation_info = tooth_data.get('orientation_info', {})
                        scale_factor = tooth_data.get('scale_factor', 250.0)
                        
                        image_record = {
                            'image_path': tooth_data.get('image_path', ''),
                            'wear_case': 'healthy',
                            'tooth_id': int(tooth_id),
                            'scale_factor': scale_factor,
                            'pixels_per_mm': scale_factor / 1000.0,
                            'orientation_info': {
                                'original_orientation': orientation_info.get('original_orientation', 'vertical'),
                                'reference_orientation': orientation_info.get('reference_orientation', 'unknown'),
                                'rotation_angle': orientation_info.get('rotation_angle', 0.0),
                                'confidence': orientation_info.get('confidence', 0.0),
                                'normalized': orientation_info.get('normalized', False),
                                'reference_aligned': orientation_info.get('reference_aligned', False)
                            },
                            'case_status': 'Healthy Baseline',
                            'tooth_status': f"Tooth {tooth_id}",
                            'wear_measurements': {
                                'area_wear_percent': 0.0,  # No wear for healthy baseline
                                'perimeter_wear_percent': 0.0,
                                'wear_score': 0.0,
                                'area_loss': 0.0,
                                'perimeter_loss': 0.0
                            }
                        }
                        per_image_data.append(image_record)
        
        except Exception as e:
            print(f"⚠️ Could not generate per_image data: {str(e)}")
        
        return per_image_data

    def regenerate_wear_measurement_results(self):
        """Regenerate wear measurement results using integrated system"""
        print("🔄 Regenerating Wear Measurement Results...")
        print("=" * 50)
        print("ℹ️ Wear measurement regeneration is now handled through the integrated menu system.")
        print("ℹ️ Please use 'Database Process (Gear Wear Analysis)' from the Picture Analysis menu.")
        print("ℹ️ This will regenerate the complete analysis using the new integrated scripts.")
        
        # Return a placeholder result indicating the new system should be used
        return {
            "status": "integrated_system",
            "message": "Wear measurement regeneration is now handled through the integrated menu system"
        }

def main():
    """Main function to run the gear wear diagnosis agent with user menu"""
    print("🔧 Comprehensive Gear Wear Diagnosis Agent")
    print("=" * 60)
    
    # Initialize the agent
    agent = GearWearDiagnosisAgent()
    
    while True:
        # Display main menu
        print("\n📋 MAIN MENU")
        print("=" * 30)
        print("1. Picture Analysis")
        print("2. Vibration Analysis")
        print("3. Diagnosis")
        print("4. Write Summary")
        print("5. View System Information")
        print("6. Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🖼️ PICTURE ANALYSIS")
                print("=" * 25)
                display_picture_analysis_menu(agent)
                    
            elif choice == "2":
                print("\n📊 VIBRATION ANALYSIS")
                print("=" * 25)
                display_vibration_analysis_menu(agent)
                    
            elif choice == "3":
                print("\n🔍 DIAGNOSIS")
                print("=" * 25)
                display_diagnosis_menu(agent)
                
            elif choice == "4":
                print("\n📝 WRITE SUMMARY")
                print("=" * 25)
                display_write_summary_menu(agent)
                
            elif choice == "5":
                display_system_information()
                
            elif choice == "6":
                print("\n👋 Thank you for using Gear Wear Diagnosis Agent!")
                print("Goodbye! 👋")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Operation cancelled by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
##old main

###############################
# from dotenv import load_dotenv
# import os
# import langchain
# import ragas



# load_dotenv()  # Load .env file

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# def Full_pipeline():
#     print("starting full pipeline")
#     # Placeholder for the full pipeline logic
#     #1.file extraction + Parsing+ chunking avg chunk size :250-500, 800 tokens for table\diagram
#     #2.metadata generation - filename, pagenumber, chunk_summary, keywords, section_type clientID\CaseID etc..
#     #3.indexing - tables to csv\markdown , tableid, pagenum, anchor saving + small text summarization of table, vector metadate to filter retrival etc
#     #4.Hybrid retrieval
#     #5.Multi document support
#     #6.gradio QA agent
#     print("pipeline ended")
    
    


# def main():
#     print("hello bitches")
#     print(OPENAI_API_KEY)
#     print(GOOGLE_API_KEY)
#     Full_pipeline()
#     return

# if __name__ == "__main__":
#     main()
    
#     #######################################