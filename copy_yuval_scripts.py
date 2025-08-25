#!/usr/bin/env python3
"""
Copy Scripts from Yuval's Branch
================================

This script copies all the scripts from Yuval's branch to the current branch.
"""

import subprocess
import os
import shutil

def run_command(cmd):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        return None

def copy_file_from_branch(branch, file_path, target_path):
    """Copy a file from a specific branch."""
    cmd = f'git show {branch}:{file_path}'
    content = run_command(cmd)
    if content:
        # Ensure target directory exists (skip for root files)
        if os.path.dirname(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Write the file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Copied {file_path} to {target_path}")
        return True
    else:
        print(f"❌ Failed to copy {file_path}")
        return False

def main():
    """Main function to copy all scripts from Yuval's branch."""
    print("🚀 Copying scripts from Yuval's branch...")
    
    # List of files to copy from Yuval's branch
    files_to_copy = [
        "Main.py",
        "app/agents.py",
        "app/chunking.py",
        "app/config.py",
        "app/eval_ragas.py",
        "app/indexing.py",
        "app/loaders.py",
        "app/metadata.py",
        "app/retrieve.py",
        "app/ui_gradio.py",
        "app/validate.py",
        "app/utils.py",
        "app/prompts.py",
        "app/logger.py",
        "app/__init__.py"
    ]
    
    success_count = 0
    total_files = len(files_to_copy)
    
    for file_path in files_to_copy:
        if copy_file_from_branch("yuval", file_path, file_path):
            success_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Successfully copied: {success_count}/{total_files} files")
    
    if success_count == total_files:
        print("✅ All scripts copied successfully!")
    else:
        print("⚠️  Some files failed to copy. Check the errors above.")

if __name__ == "__main__":
    main()
