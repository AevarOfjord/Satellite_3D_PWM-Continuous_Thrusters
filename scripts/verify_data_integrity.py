import csv
import math
import os
import sys


def verify_csv(file_path, time_col):
    print(f"Verifying {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"FAILED: File not found: {file_path}")
        return False

    with open(file_path, "r") as f:
        # potential BOM or whitespace strip
        lines = [line.strip() for line in f.readlines()]
        if not lines:
            print("FAILED: Empty file.")
            return False

        reader = csv.DictReader(lines)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames]

        rows = list(reader)

    print(f"  Rows found: {len(rows)}")
    if len(rows) == 0:
        print("FAILED: No data rows found.")
        return False

    # Check headers
    if reader.fieldnames is None or time_col not in reader.fieldnames:
        print(f"FAILED: '{time_col}' column missing. Found: {reader.fieldnames}")
        return False

    # Check monotonicity of timestamp
    prev_ts = -1.0
    for i, row in enumerate(rows):
        try:
            val_str = row[time_col]
            if not val_str:
                continue  # empty line?
            ts = float(val_str)

            # Allow equal timestamps or small floating point jitter issues,
            # but generally should increase
            # Sim starts at 0 sometimes
            if i > 0 and ts < prev_ts:
                print(f"FAILED: Timestamp not monotonic at row {i+2}: {ts} < {prev_ts}")
                return False
            prev_ts = ts

            # Quaternion check if columns exist
            if all(k in row for k in ["q0", "q1", "q2", "q3"]):
                q0, q1, q2, q3 = (
                    float(row["q0"]),
                    float(row["q1"]),
                    float(row["q2"]),
                    float(row["q3"]),
                )
                norm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
                if abs(norm - 1.0) > 1e-2:
                    print(f"WARNING: Quaternion norm deviation at row {i+2}: {norm}")

            # 3D Check
            if "Current_Z" in row:
                try:
                    z_val = float(row["Current_Z"])
                except ValueError:
                    print(f"FAILED: Invalid Z value at row {i+2}")
                    return False

        except ValueError as e:
            print(f"FAILED: Value error at row {i+2} column {time_col}: {e}")
            return False

    print("SUCCESS: Data integrity check passed.")
    return True


# Default test directory - modify as needed for your test data
base_dir = "Data/Simulation/02-01-2026_12-13-37"

files_config = [("control_data.csv", "MPC_Start_Time"), ("physics_data.csv", "Time")]

all_passed = True
for fname, t_col in files_config:
    if not verify_csv(os.path.join(base_dir, fname), t_col):
        all_passed = False

if all_passed:
    print("\nAll checks passed successfully.")
    sys.exit(0)
else:
    print("\nSome checks failed.")
    sys.exit(1)
