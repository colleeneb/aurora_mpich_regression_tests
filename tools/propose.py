#!/usr/bin/env python3
"""Draft a regression test for an issue and open a draft PR with it.

Asks the model for a reproducer and a bats file in the shape the suite already
uses, writes them into a new 1node/ or 2node/ directory, pushes a branch to
your fork, and opens a DRAFT pull request.

Nothing here runs the test: the GB10 has no Aurora MPICH. The PR body says so.

    ./propose.py issues.json ~/aurora_mpich_regression_tests --issue 7880
    ./propose.py issues.json ~/suite --issue 7880 --write-only
    ./propose.py issues.json ~/suite --issue 7880 --no-pr   # commit, no PR

Needs GITHUB_TOKEN with repo scope for the PR step, and MPICH_WATCH_API for the
model. --write-only skips both and just leaves files on disk to look at.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

FORK = "colleeneb/aurora_mpich_regression_tests"
UPSTREAM = "lowpolyneko/aurora_mpich_regression_tests"
BASE = "main"

SYSTEM = """\
You write reproducers for an MPI regression-test suite that runs on Aurora.

Given a bug report, produce two files:

1. A standalone MPI program that reproduces the failure. It must compile with
   mpicc, mpicxx or mpif90 and run under mpiexec on 1 or 2 nodes. It must print
   a definite marker on success so the bats file can grep for it. Keep it as
   small as the bug allows.
2. A bats file in exactly the style shown in the examples: a header comment
   giving the issue URL and a short prose explanation of the failure, a
   setup_file that cds to BATS_TEST_DIRNAME and compiles, a setup that cds, and
   one or more @test blocks named "mpi_issue<NNNN> <what it does>".

Reply with one JSON object and nothing else:

{"dirname": "<short_slug>_<issue number>",
 "nodes": 1|2,
 "source_name": "<filename with extension>",
 "source": "<the full program>",
 "bats": "<the full bats file>",
 "notes": "<one sentence on anything you had to guess>"}

Write only what the report supports. Where it is silent on a buffer size, a
count or a datatype, choose something plausible and say so in "notes" rather
than inventing detail that looks authoritative."""


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise SystemExit("%s failed: %s" % (" ".join(cmd), p.stderr.strip()))
    return p.stdout.strip()


def ask_model(api, model, system, prompt, timeout, temperature):
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4000,
        }).encode(),
        headers={"Content-Type": "application/json"})
    key = os.environ.get("MPICH_WATCH_KEY")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(text):
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            p = part[4:] if part.lstrip().startswith("json") else part
            if p.lstrip().startswith("{"):
                t = p
                break
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in reply: %s" % text[:200])
    return json.loads(t[i:j + 1])


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
                "bats": "test.bats" if "test.bats" in files else None,
                "lines": sum(sum(1 for _ in open(os.path.join(path, f),
                                                 errors="replace"))
                             for f in srcs),
            })
    return {"root": root, "count": len(tests), "tests": tests,
            "issues": sorted(t["issue"] for t in tests if t["issue"])}


def examples(suite, root, n=2):
    """Two whole tests, source and bats, so the model copies real style rather
    than a description of it."""
    ts = sorted((t for t in suite["tests"] if t["sources"] and t["bats"]),
                key=lambda t: t["lines"])
    picks = [ts[0], ts[len(ts) // 2]][:n]
    out = []
    for t in picks:
        d = os.path.join(root, t["dir"])
        src = t["sources"][0]
        try:
            body = open(os.path.join(d, src), errors="replace").read()
            bats = open(os.path.join(d, "test.bats"), errors="replace").read()
        except OSError:
            continue
        out.append("--- %s/%s ---\n%s\n--- %s/test.bats ---\n%s"
                   % (t["dir"], src, body, t["dir"], bats))
    return "\n\n".join(out)


def github(path, token, method="GET", payload=None):
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=json.dumps(payload).encode() if payload else None,
        method=method,
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + token,
                 "Content-Type": "application/json",
                 "User-Agent": "mpich-watch-propose"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("issues")
    ap.add_argument("suite", nargs="?",
                    default=os.environ.get("MPICH_SUITE_DIR"),
                    help="aurora_mpich_regression_tests checkout "
                         "(default: $MPICH_SUITE_DIR)")
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--repo", default=os.environ.get("MPICH_SUITE_DIR"),
                    help="local suite checkout to commit in")
    ap.add_argument("--fork", default=FORK)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--remote", default="mine",
                    help="git remote pointing at the fork")
    ap.add_argument("--api", default=os.environ.get(
        "MPICH_WATCH_API", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get(
        "MPICH_WATCH_MODEL", "local-model"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--write-only", action="store_true",
                    help="write the files and stop: no commit, no push, no PR")
    ap.add_argument("--no-pr", action="store_true",
                    help="commit and push but do not open the PR")
    args = ap.parse_args()

    if not args.suite:
        ap.error("give a suite checkout path or set MPICH_SUITE_DIR")
    suite_root = os.path.expanduser(args.suite)
    if not os.path.isdir(suite_root):
        ap.error("not a directory: %s" % suite_root)

    with open(args.issues) as f:
        issues = {i["number"]: i for i in json.load(f)["issues"]}
    suite = scan_suite(suite_root)
    if args.issue not in issues:
        raise SystemExit("issue #%d not in %s" % (args.issue, args.issues))
    if args.issue in set(suite["issues"]):
        raise SystemExit("#%d already has a test in the suite" % args.issue)

    issue = issues[args.issue]
    root = os.path.expanduser(args.repo) if args.repo else suite_root

    prompt = ("TWO EXAMPLES FROM THE SUITE\n\n%s\n\n%s\n\nISSUE #%d: %s\n%s\n\n%s"
              % (examples(suite, root), "=" * 60, issue["number"],
                 issue["title"], issue["url"], issue["body"] or "(empty)"))

    print("asking %s for a reproducer for #%d ..." % (args.model, args.issue),
          file=sys.stderr)
    draft = parse(ask_model(args.api, args.model, SYSTEM, prompt,
                            args.timeout, args.temperature))

    nodes = 1 if int(draft.get("nodes", 1)) == 1 else 2
    slug = re.sub(r"[^a-z0-9_]+", "_", str(draft["dirname"]).lower()).strip("_")
    if not slug.endswith(str(args.issue)):
        slug = "%s_%d" % (slug, args.issue)
    reldir = "%dnode/%s" % (nodes, slug)
    absdir = os.path.join(root, reldir)
    srcname = os.path.basename(str(draft["source_name"]))

    os.makedirs(absdir, exist_ok=True)
    with open(os.path.join(absdir, srcname), "w") as f:
        f.write(draft["source"].rstrip() + "\n")
    with open(os.path.join(absdir, "test.bats"), "w") as f:
        f.write(draft["bats"].rstrip() + "\n")
    print("wrote %s/{%s,test.bats}" % (reldir, srcname))
    if draft.get("notes"):
        print("model notes: %s" % draft["notes"])

    if args.write_only:
        print("\n--write-only: nothing committed. Review, then run without it.")
        return 0

    branch = "add-test-%d" % args.issue
    run(["git", "-C", root, "checkout", "-b", branch, "%s/%s"
         % (args.remote, args.base)], check=False)
    run(["git", "-C", root, "add", reldir])
    msg = ("tests: add a reproducer for #%d\n\n%s\n\nDrafted from the issue "
           "text; NOT yet run on Aurora.\n" % (args.issue, issue["title"]))
    run(["git", "-C", root, "commit", "-m", msg])
    run(["git", "-C", root, "push", "-u", args.remote, branch])
    print("pushed %s to %s" % (branch, args.remote))

    if args.no_pr:
        return 0
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("set GITHUB_TOKEN to open the PR, or pass --no-pr")

    body = (
        "Adds a reproducer for %s\n\n"
        "**This test has not been run.** It was drafted from the issue text by "
        "`mpich_watch/propose.py` on a machine with no Aurora MPICH, so it is "
        "unverified: it may not compile, and if it does it may not reproduce "
        "the failure. Run it before trusting it.\n\n"
        "- issue: %s\n- directory: `%s`\n- model: `%s`\n\n"
        "%s\n" % (issue["url"], issue["url"], reldir, args.model,
                  "Model notes: %s" % draft["notes"] if draft.get("notes")
                  else ""))
    pr = github("/repos/%s/pulls" % args.upstream, token, "POST", {
        "title": "tests: reproducer for #%d" % args.issue,
        "head": "%s:%s" % (args.fork.split("/")[0], branch),
        "base": args.base,
        "body": body,
        "draft": True,
    })
    print("draft PR: %s" % pr["html_url"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
