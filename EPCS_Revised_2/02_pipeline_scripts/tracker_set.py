"""Safe tracker update:  python tracker_set.py ID STATUS "evidence text"
STATUS is one of: DONE, "IN PROGRESS", TODO, "BLOCKED (user)".
Rewrites only the status and evidence fields of the row with that ID in build_task_tracker.py, then rebuilds the tracker."""
import ast
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("build_task_tracker.py")
NAMES = {"DONE": "D", "IN PROGRESS": "P", "TODO": "T", "BLOCKED (user)": "B"}


def main():
    tid, status, evidence = sys.argv[1], sys.argv[2], sys.argv[3]
    if status not in NAMES:
        sys.exit(f"status must be one of {list(NAMES)}")
    lines = SCRIPT.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if line.startswith(f' ("{tid}",'):
            body = line.strip().rstrip(",")
            body = re.sub(r",\s*(D|P|T|B),", lambda m: ',"@%s",' % m.group(1), body, count=1)
            row = list(ast.literal_eval(body))
            break
    else:
        sys.exit(f"id {tid} not found")

    def q(x):
        return '"' + str(x).replace("\\", "/").replace('"', "'") + '"'

    fields = [q(row[0]), q(row[1]), q(row[2]), q(row[3]), NAMES[status], q(evidence), q(row[6])]
    lines[i] = " (" + ",".join(fields) + "),"
    SCRIPT.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT)], check=True)


if __name__ == "__main__":
    main()
