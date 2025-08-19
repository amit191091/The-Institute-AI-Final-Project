import os
import numpy as np
import scipy.io as sio

# Check if h5py is available for MATLAB v7.3 files
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

class VibrationAnalysis:
    """Class for analyzing vibration data from MAT files"""
    
    def __init__(self):
        self.vibration_analysis_results = {}
    
    def analyze_vibration_data(self, mat_paths):
        """Analyze vibration data from MAT files"""
        print("📊 Analyzing vibration data...")
        
        for mat_path in mat_paths:
            if not os.path.exists(mat_path):
                print(f"⚠️ MAT file not found: {mat_path}")
                continue
                
            try:
                # Load MAT file using v7.3-safe loader
                vibration_data = self._load_mat_file_safe(mat_path)
                
                if vibration_data is None:
                    print(f"⚠️ No suitable vibration data found in MAT file: {mat_path}")
                    continue
                
                # Vibration analysis features
                analysis = {
                    'filename': os.path.basename(mat_path),
                    'data_length': len(vibration_data),
                    'rms_vibration': np.sqrt(np.mean(vibration_data**2)),
                    'peak_vibration': np.max(np.abs(vibration_data)),
                    'vibration_crest_factor': np.max(np.abs(vibration_data)) / np.sqrt(np.mean(vibration_data**2)),
                    'vibration_kurtosis': self._calculate_kurtosis(vibration_data),
                    'vibration_skewness': self._calculate_skewness(vibration_data),
                    'envelope_analysis': self._perform_envelope_analysis(vibration_data),
                    'gear_mesh_analysis': self._analyze_gear_mesh_frequency(vibration_data),
                    'bearing_analysis': self._analyze_bearing_frequencies(vibration_data)
                }
                
                self.vibration_analysis_results[mat_path] = analysis
                
            except Exception as e:
                print(f"⚠️ Error processing MAT file {mat_path}: {str(e)}")
                
        return self.vibration_analysis_results
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of the signal"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4)
    
    def _calculate_skewness(self, data):
        """Calculate skewness of the signal"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _perform_envelope_analysis(self, data, fs=1000):
        """Perform envelope analysis for bearing fault detection"""
        # Hilbert transform for envelope
        analytic_signal = np.fft.hilbert(data)
        envelope = np.abs(analytic_signal)
        
        # Envelope spectrum analysis
        envelope_fft = np.fft.fft(envelope)
        envelope_magnitude = np.abs(envelope_fft)
        frequencies = np.fft.fftfreq(len(envelope), 1/fs)
        
        # Find envelope peaks
        peak_threshold = np.max(envelope_magnitude) * 0.1
        peaks = envelope_magnitude > peak_threshold
        
        return {
            'envelope_rms': np.sqrt(np.mean(envelope**2)),
            'envelope_peak': np.max(envelope),
            'envelope_crest_factor': np.max(envelope) / np.sqrt(np.mean(envelope**2)),
            'envelope_peaks_count': np.sum(peaks)
        }
    
    def _analyze_gear_mesh_frequency(self, data, fs=1000):
        """Analyze gear mesh frequency characteristics"""
        # FFT analysis
        fft = np.fft.fft(data)
        magnitude = np.abs(fft)
        frequencies = np.fft.fftfreq(len(data), 1/fs)
        
        # Look for gear mesh frequency and harmonics
        # Assuming typical gear mesh frequency range
        mesh_freq_range = (50, 500)  # Hz
        mesh_indices = (frequencies >= mesh_freq_range[0]) & (frequencies <= mesh_freq_range[1])
        
        if np.any(mesh_indices):
            mesh_magnitude = magnitude[mesh_indices]
            mesh_frequencies = frequencies[mesh_indices]
            
            # Find dominant mesh frequency
            dominant_mesh_idx = np.argmax(mesh_magnitude)
            dominant_mesh_freq = mesh_frequencies[dominant_mesh_idx]
            
            return {
                'dominant_mesh_frequency': dominant_mesh_freq,
                'mesh_energy': np.sum(mesh_magnitude**2),
                'mesh_harmonics': self._find_harmonics(dominant_mesh_freq, frequencies, magnitude)
            }
        else:
            return {
                'dominant_mesh_frequency': None,
                'mesh_energy': 0,
                'mesh_harmonics': []
            }
    
    def _analyze_bearing_frequencies(self, data, fs=1000):
        """Analyze bearing characteristic frequencies"""
        # FFT analysis
        fft = np.fft.fft(data)
        magnitude = np.abs(fft)
        frequencies = np.fft.fftfreq(len(data), 1/fs)
        
        # Look for bearing fault frequencies (typically higher frequency)
        bearing_freq_range = (500, 2000)  # Hz
        bearing_indices = (frequencies >= bearing_freq_range[0]) & (frequencies <= bearing_freq_range[1])
        
        if np.any(bearing_indices):
            bearing_magnitude = magnitude[bearing_indices]
            bearing_frequencies = frequencies[bearing_indices]
            
            return {
                'bearing_energy': np.sum(bearing_magnitude**2),
                'bearing_peak_frequency': bearing_frequencies[np.argmax(bearing_magnitude)],
                'bearing_fault_probability': self._assess_bearing_fault(bearing_magnitude)
            }
        else:
            return {
                'bearing_energy': 0,
                'bearing_peak_frequency': None,
                'bearing_fault_probability': 'Low'
            }
    
    def _find_harmonics(self, fundamental_freq, frequencies, magnitude, num_harmonics=3):
        """Find harmonics of a fundamental frequency"""
        harmonics = []
        for i in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * i
            # Find closest frequency in the spectrum
            idx = np.argmin(np.abs(frequencies - harmonic_freq))
            if idx < len(magnitude):
                harmonics.append({
                    'harmonic_order': i,
                    'frequency': frequencies[idx],
                    'magnitude': magnitude[idx]
                })
        return harmonics
    
    def _assess_bearing_fault(self, bearing_magnitude):
        """Assess bearing fault probability based on magnitude characteristics"""
        peak_magnitude = np.max(bearing_magnitude)
        mean_magnitude = np.mean(bearing_magnitude)
        
        if peak_magnitude > 5 * mean_magnitude:
            return 'High'
        elif peak_magnitude > 2 * mean_magnitude:
            return 'Medium'
        else:
            return 'Low'
    
    def _load_mat_file_safe(self, file_path):
        """Load MAT file data safely, handling both v7.3 and older formats"""
        
        # Try h5py first for v7.3 files
        if H5PY_AVAILABLE:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Look for the main data array
                    for key in f.keys():
                        if not key.startswith('#') and not key.startswith('__'):
                            try:
                                data = f[key][()]
                                if isinstance(data, np.ndarray) and data.size > 0:
                                    if data.dtype.kind not in ['U', 'S']:  # Skip strings
                                        return data.flatten()
                            except:
                                continue
            except:
                pass  # Not a v7.3 file, try scipy.io
        
        # Since we know these are v7.3 files, if h5py failed, we can't load them
        print(f"❌ Could not load MATLAB v7.3 file: {file_path}")
        return None
    
    def generate_vibration_report(self, output_path="vibration_analysis_report.txt"):
        """Generate a vibration analysis report"""
        if not self.vibration_analysis_results:
            print("❌ No vibration analysis results available")
            return None
        
        report = []
        report.append("=" * 60)
        report.append("           VIBRATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("📊 VIBRATION ANALYSIS RESULTS")
        report.append("-" * 30)
        for path, analysis in self.vibration_analysis_results.items():
            report.append(f"File: {analysis['filename']}")
            report.append(f"  - RMS Vibration: {analysis['rms_vibration']:.3f}")
            report.append(f"  - Peak Vibration: {analysis['peak_vibration']:.3f}")
            report.append(f"  - Vibration Crest Factor: {analysis['vibration_crest_factor']:.3f}")
            report.append(f"  - Vibration Kurtosis: {analysis['vibration_kurtosis']:.3f}")
            report.append(f"  - Vibration Skewness: {analysis['vibration_skewness']:.3f}")
            
            # Gear mesh analysis
            if analysis['gear_mesh_analysis']['dominant_mesh_frequency']:
                report.append(f"  - Dominant Mesh Frequency: {analysis['gear_mesh_analysis']['dominant_mesh_frequency']:.1f} Hz")
            report.append(f"  - Mesh Energy: {analysis['gear_mesh_analysis']['mesh_energy']:.3f}")
            
            # Bearing analysis
            report.append(f"  - Bearing Fault Probability: {analysis['bearing_analysis']['bearing_fault_probability']}")
            if analysis['bearing_analysis']['bearing_peak_frequency']:
                report.append(f"  - Bearing Peak Frequency: {analysis['bearing_analysis']['bearing_peak_frequency']:.1f} Hz")
            
            # Envelope analysis
            report.append(f"  - Envelope RMS: {analysis['envelope_analysis']['envelope_rms']:.3f}")
            report.append(f"  - Envelope Crest Factor: {analysis['envelope_analysis']['envelope_crest_factor']:.3f}")
            report.append("")
        
        report.append("=" * 60)
        report.append("Report generated by Vibration Analysis System")
        report.append("=" * 60)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Vibration report saved to: {output_path}")
        return '\n'.join(report)

def main():
    """Main function to run vibration analysis"""
    print("📊 Vibration Analysis System")
    print("=" * 40)
    
    # Example usage - replace with your actual file paths
    vibration_paths = [
        "vibration_data/vibration1.mat",
        "vibration_data/vibration2.mat"
    ]
    
    # Filter existing files
    existing_vibration = [path for path in vibration_paths if os.path.exists(path)]
    
    if existing_vibration:
        analyzer = VibrationAnalysis()
        results = analyzer.analyze_vibration_data(existing_vibration)
        report = analyzer.generate_vibration_report()
        print("\n" + report)
    else:
        print("⚠️ No vibration data files found")

if __name__ == "__main__":
    main()
