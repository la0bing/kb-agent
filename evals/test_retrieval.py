import shutil
import sys

sys.path.insert(0, ".")

DB_PATH = r"evals\kbeval"
DOC_PATH = r"evals\docs\student_handbook.md"
DOC = "student_handbook.md"

shutil.rmtree(DB_PATH, ignore_errors=True)

import os
os.environ["KB_DB_PATH"] = DB_PATH

from kb_agent import store

passed = 0
total = 0


def check(name, ok, detail):
    global passed, total
    total += 1
    if ok:
        passed += 1
    print(("PASS  " if ok else "FAIL  ") + name + " (" + detail + ")")


report = store.ingest_file(DOC_PATH, False)
print("ingested " + report["source"] + " -> " + str(report["chunks"]) + " chunks")
print("")

hits = store.search("what is the minimum attendance to sit for the final exam", "", 5)
check("test 1 attendance rule is first", hits[0]["source"] == DOC and "80%" in hits[0]["text"], str(hits[0]["score"]))

hits = store.search("what happens if my cgpa is too low", "", 5)
check("test 2 correct section is first", "academic probation" in hits[0]["text"], hits[0]["headings"])

hits = store.search("how do I claim mileage for driving to work", "", 5)
check("test 3 off topic question returns nothing", len(hits) == 0, str(len(hits)) + " hits")

print("")
print(str(passed) + "/" + str(total) + " passed")
shutil.rmtree(DB_PATH, ignore_errors=True)
sys.exit(0 if passed == total else 1)
