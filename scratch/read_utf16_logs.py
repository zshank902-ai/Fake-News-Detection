import os

def read_log_file(filepath):
    encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8', 'latin-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read(), enc
        except Exception:
            continue
    return None, None

def print_full_logs():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_logs = [
        'training_gpu_5epochs_log.txt',
        'training_super_sota_final_log.txt',
        'training_gpu_final_v3_log.txt'
    ]
    
    for log in target_logs:
        path = os.path.join(parent_dir, log)
        if os.path.exists(path):
            content, enc = read_log_file(path)
            print("\n" + "=" * 85)
            print(f"FULL LOG: {log} (Encoding: {enc})")
            print("=" * 85)
            print(content)
        else:
            print(f"Log not found: {log}")

print_full_logs()
