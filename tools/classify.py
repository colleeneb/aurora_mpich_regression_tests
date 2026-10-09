#!/usr/bin/env python3
"""Ask a model: can this issue become a 1- or 2-node test in our suite?

Takes watch.py output and a suite checkout. Shows the model a few real tests
from the suite so it judges against what we actually write, not a generic idea
of a test, then asks the one question for each issue.

    ./watch.py --state open > issues.json
    ./classify.py issues.json ~/aurora_mpich_regression_tests

    --eval        score against the suite: issues with a test should be yes
    --dry-run     print the prompt for one issue and exit
    --only N      just issue #N

Speaks OpenAI-compatible /v1/chat/completions, so vLLM and llama-server both
work unchanged:

    vllm serve <model> --port 8000
    llama-server -m <model>.gguf --port 8000
    export MPICH_WATCH_API=http://localhost:8000/v1
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

QUESTION = """\
You triage MPI bug reports for a regression-test suite that runs on Aurora.

Each test is one directory under 1node/ or 2node/, holding a small standalone
program and a bats file that compiles it with mpicc/mpif90 and runs it under
mpiexec. Examples from the suite are shown below.

For the issue given, answer one question: could someone write a test of that
shape from this report? Judge the report as written. If it describes a failure
needing many nodes, a large proprietary application, or no definite pass/fail,
the answer is no.

Reply with one JSON object and nothing else:

{"usable": true|false,
 "nodes": 1|2|null,
 "confidence": "high"|"medium"|"low",
 "reason": "<one sentence>",
 "quotes": ["<text copied verbatim from the issue>", ...]}

Every string in "quotes" must be copied character for character from the issue.
Do not paraphrase, do not invent. Quote what the answer rests on: the scale, the
failing call, the pass/fail signal. If the issue does not say, use low
confidence and say so in "reason"."""


# Test directories are named <description>_<issue number>, e.g.
# ipc_cache_evict_7947. That suffix is the only link between a test and the
# issue it guards.
NUM_RE = re.compile(r"_(\d{3,6})$")
SRC_EXT = (".c", ".cc", ".cpp", ".cxx", ".f", ".f90", ".F90", ".hip", ".cu",
           ".sh")


def scan_suite(root):
    """Walk 1node/ and 2node/ for the tests that already exist."""
    tests = []
    for sub in ("1node", "2node"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            if not os.path.isdir(path):
                continue
            m = NUM_RE.search(name)
            files = sorted(os.listdir(path))
            srcs = [f for f in files if f.endswith(SRC_EXT)]
            tests.append({
                "dir": "%s/%s" % (sub, name),
                "issue": int(m.group(1)) if m else None,
                "nodes": 1 if sub == "1node" else 2,
                "sources": srcs,
                "lines": sum(sum(1 for _ in open(os.path.join(path, f),
                                                 errors="replace"))
                             for f in srcs),
            })
    return {"root": root, "count": len(tests), "tests": tests,
            "issues": sorted(t["issue"] for t in tests if t["issue"])}


def suite_examples(suite, n=6):
    """A spread of real tests: smallest, largest, and some between, so the
    model sees that 6 lines and 224 lines are both normal here."""
    ts = sorted((t for t in suite["tests"] if t["sources"]),
                key=lambda t: t["lines"])
    if len(ts) <= n:
        picks = ts
    else:
        step = (len(ts) - 1) / (n - 1)
        picks = [ts[round(i * step)] for i in range(n)]
    return "\n".join(
        "  %-42s %d node, %s, %d source lines"
        % (t["dir"], t["nodes"], ", ".join(t["sources"]), t["lines"])
        for t in picks)


def build_prompt(issue, suite):
    return ("TESTS ALREADY IN THE SUITE (%d total)\n%s\n\n"
            "ISSUE #%d: %s\nState: %s, opened %s\n\n%s"
            % (suite["count"], suite_examples(suite),
               issue["number"], issue["title"], issue["state"],
               issue["created"][:10], issue["body"] or "(empty body)"))


def ask(api, model, prompt, timeout, temperature):
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": QUESTION},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 600,
        }).encode(),
        headers={"Content-Type": "application/json"})
    key = os.environ.get("MPICH_WATCH_KEY")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(text):
    """Models wrap JSON in fences or prose often enough to handle it here."""
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            p = part[4:] if part.lstrip().startswith("json") else part
            if p.lstrip().startswith("{"):
                t = p
                break
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in reply: %s" % text[:120])
    return json.loads(t[i:j + 1])


def bad_quotes(verdict, body):
    """A quote not in the body means the model made it up. Whitespace is
    normalised; nothing else is forgiven."""
    norm = " ".join((body or "").split())
    return [q for q in (verdict.get("quotes") or [])
            if " ".join(str(q).split()) not in norm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("issues", help="watch.py output")
    ap.add_argument("suite", nargs="?",
                    default=os.environ.get("MPICH_SUITE_DIR"),
                    help="aurora_mpich_regression_tests checkout "
                         "(default: $MPICH_SUITE_DIR)")
    ap.add_argument("--api", default=os.environ.get(
        "MPICH_WATCH_API", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get(
        "MPICH_WATCH_MODEL", "local-model"))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only", type=int, metavar="N")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval", action="store_true",
                    help="score against the suite: an issue with a test "
                         "should come back usable")
    ap.add_argument("-o", "--out", help="also write verdicts as JSON")
    args = ap.parse_args()

    if not args.suite:
        ap.error("give a suite checkout path or set MPICH_SUITE_DIR")
    root = os.path.expanduser(args.suite)
    if not os.path.isdir(root):
        ap.error("not a directory: %s" % root)

    with open(args.issues) as f:
        issues = json.load(f)["issues"]
    suite = scan_suite(root)
    tested = set(suite["issues"])

    if args.only:
        issues = [i for i in issues if i["number"] == args.only]
    if not issues:
        print("no issues to classify", file=sys.stderr)
        return 1

    if args.dry_run:
        print(QUESTION)
        print("\n" + "=" * 72 + "\n")
        print(build_prompt(issues[0], suite))
        return 0

    results = []
    for i in issues:
        row = {"number": i["number"], "title": i["title"], "url": i["url"],
               "state": i["state"], "has_test": i["number"] in tested}
        try:
            v = parse(ask(args.api, args.model, build_prompt(i, suite),
                          args.timeout, args.temperature))
            row["verdict"] = v
            row["bad_quotes"] = bad_quotes(v, i["body"])
        except (urllib.error.URLError, ValueError, KeyError,
                json.JSONDecodeError) as e:
            row["error"] = "%s: %s" % (type(e).__name__, e)
        results.append(row)

        if "error" in row:
            print("#%-6d ERROR  %s" % (i["number"], row["error"]))
            continue
        v = row["verdict"]
        print("#%-6d %-4s %-7s %-7s %s%s"
              % (i["number"],
                 "YES" if v.get("usable") else "no",
                 "%sn" % v.get("nodes") if v.get("nodes") else "-",
                 v.get("confidence", "?"),
                 (v.get("reason") or "")[:52],
                 "   [!] %d invented quote(s)" % len(row["bad_quotes"])
                 if row["bad_quotes"] else ""))

    if args.eval:
        scored = [r for r in results if "verdict" in r]
        tp = [r for r in scored if r["has_test"] and r["verdict"].get("usable")]
        fn = [r for r in scored if r["has_test"]
              and not r["verdict"].get("usable")]
        flagged = [r for r in scored if not r["has_test"]
                   and r["verdict"].get("usable")]
        made_up = [r for r in scored if r["bad_quotes"]]
        print("\n%s\nEVAL over %d issue(s)\n%s" % ("=" * 72, len(scored), "=" * 72))
        print("  has a test, called usable   : %d" % len(tp))
        print("  has a test, called NOT usable: %d   <- misses" % len(fn))
        for r in fn:
            print("      #%-6d %s" % (r["number"], r["verdict"].get("reason", "")[:60]))
        print("  no test, called usable       : %d   (candidates, or false alarms)"
              % len(flagged))
        for r in flagged:
            print("      #%-6d %s" % (r["number"], r["title"][:60]))
        print("  verdicts with invented quotes: %d" % len(made_up))
        # An issue that already has a test is the clearest ground truth there
        # is: someone wrote one, so it was possible.
        if tp or fn:
            print("\n  recall on known-good: %d/%d" % (len(tp), len(tp) + len(fn)))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=1)
        print("\nwrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
