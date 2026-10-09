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
 "launch_lines": ["<every mpiexec/mpirun/srun line in the issue, verbatim>"],
 "nodes": 1|2|null,
 "confidence": "high"|"medium"|"low",
 "reason": "<one sentence>",
 "quotes": ["<text copied verbatim from the issue>", ...]}

Fill "launch_lines" first. Search the whole issue for lines that run the
program -- mpiexec, mpirun or srun -- and list EVERY one, copied exactly,
including any environment variables before the command. An issue often shows
several: different scales, or the same run with different settings. Use an
empty list only if the issue contains none.

Copy only what the line actually contains. If a line is longer than about 120
characters, copy the first 120 and stop: the part that matters is the command
and its -n / -ppn / -N flags, not a long --cpu-bind list. Never write a launch
line that is not in the issue. If the issue describes a scale only in prose,
leave it out of this list and say so in "reason" instead.

Getting "launch_lines" complete and exact matters more than anything else you
report: the node count is computed from those lines, not from your answer.

Also give "nodes" as your own reading of the smallest scale that still
reproduces the failure, in nodes rather than ranks. If the issue has no launch
line, that reading is all there is; otherwise it is a cross-check.

On "usable", judge scale by the smallest run the issue says still fails. A
report that fails at 16 nodes and also at 2 is usable. Answer usable=false on
scale grounds only when every run it describes needs more than 2 nodes.

Every string in "quotes" must be copied character for character from the issue.
Do not paraphrase, do not invent. Quote what the answer rests on: the scale, the
failing call, the pass/fail signal. If the issue does not say, use low
confidence and say so in "reason"."""


# Nodes from a launch line. The model finds the lines; this does the division,
# because an 8B model reliably extracts "mpiexec -n 2 -ppn 1" and then just as
# reliably answers 1. Arithmetic belongs in code.
RANKS_RE = re.compile(r"(?:^|\s)-(?:n|np)\s+(\d+)\b")
PPN_RE = re.compile(r"(?:^|\s)--?ppn\s+(\d+)\b"
                    r"|(?:^|\s)--ntasks-per-node[= ](\d+)\b")
NODES_FLAG_RE = re.compile(r"(?:^|\s)(?:-N|--nodes)[= ](\d+)\b")


def nodes_in(line):
    """Node count for one launch line, or None if it does not say.

    -N / --nodes wins outright. Otherwise nodes = ranks / ranks-per-node,
    rounded up. Ranks alone say nothing about nodes, so "-n 4" with no -ppn
    is read as one node.
    """
    m = NODES_FLAG_RE.search(line)
    if m:
        return int(m.group(1))
    mr = RANKS_RE.search(line)
    if not mr:
        return None
    mp = PPN_RE.search(line)
    if not mp:
        return 1
    ppn = int(mp.group(1) or mp.group(2))
    return -(-int(mr.group(1)) // ppn) if ppn else None


def smallest_nodes(lines):
    """The smallest scale the issue demonstrates, which is the one that
    decides whether a 1node/ or 2node/ test is possible."""
    counts = [n for n in (nodes_in(l) for l in lines or []) if n]
    return min(counts) if counts else None


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
            "max_tokens": 2000,
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
    """Anything the model claims to have copied, checked against the issue.
    An invented launch line is worse than an invented quote: the node count is
    computed from it. Whitespace is normalised; nothing else is forgiven."""
    norm = " ".join((body or "").split())
    bad = []
    for q in list(verdict.get("quotes") or []) + \
            list(verdict.get("launch_lines") or []):
        t = " ".join(str(q).split())
        # A line the model was told to cut at 120 characters will not match
        # whole, so a long-enough prefix counts as found.
        if t in norm or (len(t) >= 40 and t[:120] in norm):
            continue
        bad.append(q)
    return bad


# Issues that should come back usable=false. Chosen because each is out of
# scope for a reason stated in its own title or body, not because a model got
# them wrong: collectives that only fail at 8 to 512 nodes, performance
# variance with no pass/fail, and one documentation request.
#
# Without these, --eval cannot tell a working classifier from one that answers
# "usable" to everything: recall alone is 100% for both.
KNOWN_BAD = {
    7983: "128+ nodes, proprietary application",
    7971: "32 nodes, 96 ppn",
    7645: "256 nodes, performance",
    7609: "8+ nodes, 96 ppn",
    7608: "32 nodes, 96 ppn",
    7606: "8+ nodes, 96 ppn",
    7604: "performance variance, no pass/fail",
    7603: "32 nodes, variance only",
    7593: "64 nodes",
    7569: "8+ nodes, 96 ppn",
    7568: "8+ nodes, 96 ppn",
    7365: "documentation request, not a bug",
    7330: "performance variance, no pass/fail",
    7118: "multiple nodes, no scale given",
}


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
            # Trust only the launch lines that are really in the issue: a
            # fabricated one would otherwise feed the arithmetic.
            real = [l for l in (v.get("launch_lines") or [])
                    if l not in row["bad_quotes"]]
            row["computed_nodes"] = smallest_nodes(real)
            row["model_nodes"] = v.get("nodes")
        except (urllib.error.URLError, ValueError, KeyError,
                json.JSONDecodeError) as e:
            row["error"] = "%s: %s" % (type(e).__name__, e)
        results.append(row)

        if "error" in row:
            print("#%-6d ERROR  %s" % (i["number"], row["error"]))
            continue
        v = row["verdict"]
        cn = row.get("computed_nodes")
        print("#%-6d %-4s %-7s %-7s %s%s"
              % (i["number"],
                 "YES" if v.get("usable") else "no",
                 "%sn" % cn if cn else "-",
                 v.get("confidence", "?"),
                 (v.get("reason") or "")[:52],
                 "   [!] %d invented quote(s)" % len(row["bad_quotes"])
                 if row["bad_quotes"] else ""))
        for ln in (v.get("launch_lines") or [])[:4]:
            flag = "!" if ln in row["bad_quotes"] else " "
            n = nodes_in(ln)
            print("        %s %-5s %s"
                  % (flag, "%dn" % n if n else "-", str(ln)[:62]))
        if cn and row.get("model_nodes") and cn != row["model_nodes"]:
            print("          (model read it as %sn; computed from the lines "
                  "above: %sn)" % (row["model_nodes"], cn))

    if args.eval:
        scored = [r for r in results if "verdict" in r]
        usable = lambda r: bool(r["verdict"].get("usable"))

        # Positives: an issue that already has a test is proof a test was
        # possible, so the classifier should say so.
        tp = [r for r in scored if r["has_test"] and usable(r)]
        fn = [r for r in scored if r["has_test"] and not usable(r)]

        # Negatives: issues ruled out by their own text. Needed because
        # recall alone cannot fail -- a classifier stuck on "yes" scores 100%.
        neg = [r for r in scored if r["number"] in KNOWN_BAD]
        tn = [r for r in neg if not usable(r)]
        fp = [r for r in neg if usable(r)]

        other = [r for r in scored if not r["has_test"]
                 and r["number"] not in KNOWN_BAD and usable(r)]
        made_up = [r for r in scored if r["bad_quotes"]]

        print("\n%s\nEVAL over %d issue(s)\n%s"
              % ("=" * 72, len(scored), "=" * 72))

        print("\nKNOWN GOOD -- %d issue(s) that already have a test" % (len(tp) + len(fn)))
        print("  called usable    : %d" % len(tp))
        print("  called NOT usable: %d   <- misses" % len(fn))
        for r in fn:
            print("      #%-6d %s" % (r["number"],
                                      r["verdict"].get("reason", "")[:58]))

        print("\nKNOWN BAD -- %d issue(s) ruled out by their own text" % len(neg))
        print("  called NOT usable: %d   <- correct" % len(tn))
        print("  called usable    : %d   <- false alarms" % len(fp))
        for r in fp:
            print("      #%-6d %-34s (%s)"
                  % (r["number"], r["title"][:34], KNOWN_BAD[r["number"]]))

        print("\nUNLABELLED -- %d called usable, for a human to judge" % len(other))
        for r in other[:12]:
            print("      #%-6d %s" % (r["number"], r["title"][:58]))
        if len(other) > 12:
            print("      ... and %d more" % (len(other) - 12))

        print("\n  verdicts with invented quotes: %d/%d"
              % (len(made_up), len(scored)))
        if tp or fn:
            print("  recall    (known good called usable)    : %d/%d"
                  % (len(tp), len(tp) + len(fn)))
        if neg:
            print("  specificity (known bad called not usable): %d/%d"
                  % (len(tn), len(neg)))
            if not tn:
                print("\n  Specificity 0 means the classifier never says no, so its")
                print("  recall is meaningless. Fix that before reading anything else.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=1)
        print("\nwrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
