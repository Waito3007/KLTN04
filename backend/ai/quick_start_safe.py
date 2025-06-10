"""
Windows-Safe Quick Start Script
Automated setup for Multimodal Fusion Model with proper Windows encoding support
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import locale
    import codecs
    # Force console encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# Simple logging without emoji to avoid encoding issues
class SimpleLogger:
    def __init__(self):
        self.log_file = Path(__file__).parent / 'quick_start_safe.log'
        
    def log(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)
        
        # Write to log file with UTF-8 encoding
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except:
            pass  # Ignore log file errors
    
    def info(self, message):
        self.log("INFO", message)
    
    def error(self, message):
        self.log("ERROR", message)
    
    def success(self, message):
        self.log("SUCCESS", message)

logger = SimpleLogger()

class QuickStartSetup:
    """Windows-safe automated setup class"""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.setup_status = {
            'environment': False,
            'dependencies': False,
            'nltk': False,
            'dataset': False,
            'model_test': False,
            'training_test': False
        }
        
    def run_command(self, command, description):
        """Run shell command with Windows-safe encoding"""
        logger.info(f"Running {description}...")
        try:
            # Use system default encoding for subprocess on Windows
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8' if sys.platform != "win32" else None
            )
            
            if result.returncode == 0:
                logger.success(f"{description} completed successfully")
                return True
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"{description} failed: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"{description} error: {str(e)}")
            return False
    
    def check_python_version(self):
        """Check Python version"""
        logger.info("Checking Python version...")
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            logger.success(f"Python {version.major}.{version.minor}.{version.micro} detected")
            return True
        else:
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
    
    def setup_environment(self):
        """Setup virtual environment if needed"""
        logger.info("Checking virtual environment...")
        
        # Check if already in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.success("Virtual environment detected")
            self.setup_status['environment'] = True
            return True
        
        # Check if venv exists
        venv_path = self.current_dir / 'venv'
        if venv_path.exists():
            logger.info("Virtual environment found")
            self.setup_status['environment'] = True
            return True
        
        # Create virtual environment
        logger.info("Creating virtual environment...")
        if self.run_command(f'python -m venv "{venv_path}"', "Create virtual environment"):
            logger.success("Virtual environment created")
            logger.info("Please activate the virtual environment and run this script again")
            logger.info(f"Windows: {venv_path}\\Scripts\\activate")
            logger.info(f"Linux/Mac: source {venv_path}/bin/activate")
            self.setup_status['environment'] = True
            return True
        
        return False
    
    def install_dependencies(self):
        """Install required packages"""
        logger.info("Installing dependencies...")
        
        packages = [
            'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
            'transformers',
            'scikit-learn',
            'pandas',
            'numpy',
            'nltk',
            'tqdm'
        ]
        
        success_count = 0
        for package in packages:
            if self.run_command(f'pip install {package}', f"Install {package.split()[0]}"):
                success_count += 1
        
        if success_count >= len(packages) - 1:  # Allow 1 failure
            logger.success("Dependencies installed successfully")
            self.setup_status['dependencies'] = True
            return True
        else:
            logger.error("Too many dependency installation failures")
            return False
    
    def setup_nltk(self):
        """Download NLTK data"""
        logger.info("Setting up NLTK data...")
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            logger.success("NLTK data downloaded")
            self.setup_status['nltk'] = True
            return True
        except Exception as e:
            logger.error(f"NLTK setup failed: {str(e)}")
            return False
    
    def check_dataset(self):
        """Check if dataset exists"""
        logger.info("Checking dataset...")
        dataset_path = self.current_dir / 'training_data' / 'improved_100k_multimodal_training.json'
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.success(f"Dataset found with {len(data)} samples")
                    self.setup_status['dataset'] = True
                    return True
            except Exception as e:
                logger.error(f"Dataset file corrupted: {str(e)}")
                return False
        else:
            logger.error("Dataset not found. Please ensure the training data is available.")
            return False
    
    def test_model_components(self):
        """Test model components"""
        logger.info("Testing model components...")
        try:
            # Test imports
            import torch
            import transformers
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            logger.success("All imports successful")
            
            # Test basic functionality
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Device: {device}")
            
            self.setup_status['model_test'] = True
            return True
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False
    
    def run_quick_training_test(self):
        """Run quick training test"""
        logger.info("Running quick training test...")
        test_script = self.current_dir / 'test_windows_compatible.py'
        
        if test_script.exists():
            if self.run_command(f'python "{test_script}"', "Quick training test"):
                logger.success("Training test completed")
                self.setup_status['training_test'] = True
                return True
        else:
            logger.error("Training test script not found")
        
        return False
    
    def generate_report(self):
        """Generate setup report"""
        logger.info("Generating setup report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'setup_status': self.setup_status,
            'success_rate': sum(self.setup_status.values()) / len(self.setup_status) * 100,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'recommendations': []
        }
        
        # Add recommendations
        if not self.setup_status['dependencies']:
            report['recommendations'].append("Install missing dependencies manually")
        if not self.setup_status['dataset']:
            report['recommendations'].append("Download or generate training dataset")
        if not self.setup_status['training_test']:
            report['recommendations'].append("Run training test manually to verify setup")
        
        # Save report
        report_path = self.current_dir / 'setup_report.json'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.success(f"Setup report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
        
        return report
    
    def run_full_setup(self):
        """Run complete setup process"""
        logger.info("Starting Quick Start Setup for Multimodal Fusion Model")
        logger.info("=" * 50)
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Environment Setup", self.setup_environment),
            ("Dependencies Installation", self.install_dependencies),
            ("NLTK Setup", self.setup_nltk),
            ("Dataset Check", self.check_dataset),
            ("Model Components Test", self.test_model_components),
            ("Training Test", self.run_quick_training_test)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"STEP: {step_name}")
            try:
                success = step_func()
                if success:
                    logger.success(f"{step_name} completed")
                else:
                    logger.error(f"{step_name} failed")
            except Exception as e:
                logger.error(f"{step_name} error: {str(e)}")
            
            logger.info("-" * 30)
        
        # Generate final report
        report = self.generate_report()
        
        logger.info("=" * 50)
        logger.info("SETUP COMPLETE")
        logger.info(f"Success Rate: {report['success_rate']:.1f}%")
        
        if report['success_rate'] >= 80:
            logger.success("Setup successful! You can now run the training script.")
            logger.info("Next steps:")
            logger.info("1. python train_enhanced_100k_fixed.py")
            logger.info("2. python evaluate_multimodal_model.py")
        else:
            logger.error("Setup incomplete. Please check the issues above.")
            if report['recommendations']:
                logger.info("Recommendations:")
                for rec in report['recommendations']:
                    logger.info(f"- {rec}")
        
        return report

def main():
    """Main function"""
    try:
        setup = QuickStartSetup()
        report = setup.run_full_setup()
        return report['success_rate'] >= 80
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Setup failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
