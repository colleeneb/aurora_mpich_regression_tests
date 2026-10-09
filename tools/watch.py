#!/usr/bin/env python3
"""Dump aurora-labelled pmodels/mpich issues as JSON.

Fetches and prints. No filtering, no heuristics, no state: whoever consumes
this decides what matters.

  ./watch.py                      # every aurora issue, to stdout
  ./watch.py --state open         # only open ones
  ./watch.py --since 2026-07-01   # created on or after a date
  ./watch.py -o issues.json       # to a file instead of stdout

Reads public issues anonymously; no token, no credentials. GitHub allows 10
unauthenticated searches a minute, which is far more than this needs.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request

REPO = "pmodels/mpich"
LABEL = "aurora"


def api(url):
    """Anonymous GET. Public issues need no credentials, so none are read:
    there is nothing here that could leak a token."""
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "mpich-aurora-watch",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch(repo, label, state, since):
    q = "repo:%s label:%s is:issue" % (repo, label)
    if state in ("open", "closed"):
        q += " is:%s" % state
    if since:
        q += " created:>=%s" % since

    out, page = [], 1
    while True:
        d = api("https://api.github.com/search/issues?q=%s&sort=created"
                "&order=desc&per_page=100&page=%d"
                % (urllib.parse.quote(q), page))
        items = d.get("items", [])
        out.extend(items)
        # The search API caps out at 1000 results; nothing here comes close.
        if len(items) < 100 or len(out) >= d.get("total_count", 0):
            break
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--state", default="all", choices=("all", "open", "closed"))
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="only issues created on or after this date")
    ap.add_argument("-o", "--out", default="-", help='output file, or "-"')
    args = ap.parse_args()

    issues = [{
        "number": i["number"],
        "title": i["title"],
        "url": i["html_url"],
        "state": i["state"],
        "created": i["created_at"],
        "updated": i["updated_at"],
        "closed": i.get("closed_at"),
        "user": (i.get("user") or {}).get("login"),
        "labels": [l["name"] for l in i.get("labels", [])],
        "comments": i.get("comments", 0),
        "body": i.get("body") or "",
    } for i in fetch(args.repo, args.label, args.state, args.since)]

    payload = {"repo": args.repo, "label": args.label,
               "state": args.state, "count": len(issues), "issues": issues}

    if args.out == "-":
        json.dump(payload, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print("wrote %d issue(s) to %s" % (len(issues), args.out),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
