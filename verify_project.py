#!/usr/bin/env python
"""
Verification Script - Check All Required Files
"""

import os
from pathlib import Path

def verify_project():
    """Verify all required files exist"""

    required_files = {
        'Main Scripts': [
            'sac_all_in_one.py',
            'main.py',
            'visualize_standalone.py'
        ],
        'Source Files': [
            'src/__init__.py',
            'src/train_sac.py',
            'src/evaluate.py',
            'src/compare_passive.py',
            'src/visualize.py'
        ],
        'Environment': [
            'src/env/suspension_env.py'
        ],
        'Utilities': [
            'src/utils/rewards.py',
            'src/utils/logger.py',
            'src/utils/metrics.py',
        ],
        'Scripts': [
            'scripts/run_experiment.sh',
            'scripts/test_rewards.sh'
        ],
        'Documentation': [
            'requirements.txt'
        ]
    }

    print("="*80)
    print("SAC PROJECT VERIFICATION")
    print("="*80)

    project_root = Path(__file__).parent
    all_exist = True
    total_files = 0
    found_files = 0

    for category, files in required_files.items():
        print(f"\n📁 {category}:")
        for file_path in files:
            total_files += 1
            full_path = project_root / file_path
            exists = full_path.exists()

            if exists:
                size = full_path.stat().st_size
                lines = 0
                if full_path.suffix in ['.py', '.sh', '.md', '.txt']:
                    try:
                        with open(full_path, 'r') as f:
                            lines = len(f.readlines())
                    except:
                        pass

                status = "✓"
                found_files += 1
                info = f"({size:,} bytes"
                if lines > 0:
                    info += f", {lines} lines"
                info += ")"
            else:
                status = "✗ MISSING"
                info = ""
                all_exist = False

            print(f"  {status} {file_path} {info}")

    print("\n" + "="*80)
    print(f"SUMMARY: {found_files}/{total_files} files found")

    if all_exist:
        print("✅ ALL FILES PRESENT!")
        print("\nYou can start training with:")
        print("  python sac_all_in_one.py --mode full --gpu 4")
        print("  OR")
        print("  cd scripts && ./run_experiment.sh")
    else:
        print("❌ SOME FILES MISSING")
        print("\nMissing files need to be created.")

    print("="*80)

    return all_exist

if __name__ == '__main__':
    verify_project()
