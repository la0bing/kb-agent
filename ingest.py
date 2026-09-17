import os
import sys

from kb_agent import store

# document titles often carry characters the default Windows console cannot print
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_topics = "--no-topics" not in sys.argv

    root = args[0] if args else "docs"
    if os.path.isfile(root):
        targets = [root]
    else:
        targets = []
        for folder, _, names in os.walk(root):
            targets.extend(os.path.join(folder, n) for n in sorted(names) if store.supported(n))

    if not targets:
        print("nothing to ingest at " + root)
        return

    print("found " + str(len(targets)) + " files")
    total = 0
    for i, target in enumerate(targets, 1):
        try:
            result = store.ingest_file(target, with_topics)
            total += result["chunks"]
            print("[%d/%d] %s -> %d chunks [%s]" % (i, len(targets), result["source"], result["chunks"], result["topic"] or "no topic"))
        except Exception as error:
            print("[%d/%d] %s FAILED: %s" % (i, len(targets), os.path.basename(target), error))

    print("done, " + str(total) + " chunks in " + store.DB_PATH)


if __name__ == "__main__":
    main()
