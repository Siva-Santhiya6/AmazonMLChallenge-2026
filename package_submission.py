import zipfile
import os

zip_path = 'team_resolver_submission.zip'
if os.path.exists(zip_path):
    os.remove(zip_path)

print("Creating submission package:", zip_path, "...")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # 1. Output files
    print("  Adding output files...")
    zf.write('output/matching_results.tsv', 'output/matching_results.tsv')
    zf.write('output/candidate_pairs.tsv', 'output/candidate_pairs.tsv')
    if os.path.exists('output/tuning_results.csv'):
        zf.write('output/tuning_results.csv', 'output/tuning_results.csv')
    
    # 2. Code directory
    print("  Adding code directory...")
    code_dir = 'code/business_entity_resolution'
    for root, dirs, files in os.walk(code_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            arcname = os.path.relpath(p, '.')
            zf.write(p, arcname)
            
    # 3. Model file
    if os.path.exists('model_optimized.joblib'):
        zf.write('model_optimized.joblib', 'code/business_entity_resolution/model_optimized.joblib')
        
    # 4. Documentation
    print("  Adding documentation...")
    if os.path.exists('Documentation_template.md'):
        zf.write('Documentation_template.md', 'Documentation_template.md')
    if os.path.exists('README.md'):
        zf.write('README.md', 'README.md')
        
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\nSUCCESS: Submission package created -> {zip_path} ({size_mb:.2f} MB)")
