"""Publish a web page as a tab on the live demo page in salt (/p/nanogpt-live).

  python3 demo/tab.py publish demo/tabs/monitor.html --title "Training monitor"
  python3 demo/tab.py list
  python3 demo/tab.py delete "Training monitor"
  python3 demo/tab.py clear          # every tab on the page, and the files in demo/tabs/

A tab is one HTML file: markup, an optional <style> block, and <script> blocks, which the page shows
in its JavaScript pane. Publishing under a title that already exists replaces that tab's code.
`publish` then waits for the page to run the code and prints any error it hit, so a mistake can be
fixed and published again. It can only report if the demo page is open in a browser.

Environment:
    SALT_URL    where salt is served (default http://localhost:8080)
    SALT_TOKEN  a bearer token for an admin user. Left unset, one is taken from the local dev stack
                by running `sli agent auth` in SALT_DIR (default ~/Desktop/salt).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SALT_URL = os.environ.get("SALT_URL", "http://localhost:8080").rstrip("/")
SALT_DIR = os.path.expanduser(os.environ.get("SALT_DIR", "~/Desktop/salt"))
API = "/api/p/nanogpt-live"
REPORT_WAIT_SECONDS = 12
INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def token():
    if os.environ.get("SALT_TOKEN"):
        return os.environ["SALT_TOKEN"]
    result = subprocess.run([os.path.join(SALT_DIR, "sli"), "agent", "auth", "--format", "json"],
                            capture_output=True, text=True, cwd=SALT_DIR)
    if result.returncode != 0:
        sys.exit(f"Could not get a token from the local salt stack: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)["access_token"]


def call(method, path, body=None, _token=[]):
    if not _token:
        _token.append(token())
    request = urllib.request.Request(SALT_URL + API + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {_token[0]}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"salt answered {error.code}: {error.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as error:
        sys.exit(f"Could not reach salt at {SALT_URL}: {error.reason}")


def tabs():
    return call("GET", "/state")["tabs"]


def split_page(source):
    """The page's inline scripts, joined, and the page without them."""
    scripts = [match.strip("\n") for match in INLINE_SCRIPT.findall(source)]
    return INLINE_SCRIPT.sub("", source).strip() + "\n", "\n\n".join(scripts) + ("\n" if scripts else "")


def find(title_or_id):
    matches = [tab for tab in tabs() if title_or_id in (tab["id"], tab["title"])]
    if not matches:
        sys.exit(f"There is no tab called {title_or_id!r}. Run `list` to see the tabs.")
    return matches[0]


def wait_for_report(tab):
    """What happened when the page ran this revision of the tab, or None if no page told us."""
    deadline = time.time() + REPORT_WAIT_SECONDS
    while time.time() < deadline:
        report = call("GET", f"/tabs/report?id={tab['id']}")["report"]
        if report and report["revision"] == tab["revision"]:
            return report
        time.sleep(1)
    return None


def publish(path, title, author):
    with open(path) as f:
        html, js = split_page(f.read())
    existing = [tab for tab in tabs() if tab["title"] == title]
    saved = call("POST", "/tabs", dict(id=existing[0]["id"] if existing else "", title=title, html=html, js=js, author=author))["tab"]
    print(f"{'Replaced' if existing else 'Added'} the tab {title!r} ({len(html)} characters of HTML, {len(js)} of JavaScript).")
    report = wait_for_report(saved)
    if report is None:
        print("No browser reported back, so it is not known whether the code runs. Is the demo page open?")
    elif report["errors"]:
        print("The page ran it and hit these errors:")
        for error in report["errors"]:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("The page ran it without errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    publish_parser = commands.add_parser("publish", help="add a tab, or replace the tab with this title")
    publish_parser.add_argument("file")
    publish_parser.add_argument("--title", required=True)
    publish_parser.add_argument("--author", choices=["ai", "human"], default="ai",
                                help="'ai' tabs are typed into the page's editor before they open (default)")
    commands.add_parser("list", help="the tabs on the page")
    commands.add_parser("delete", help="remove one tab").add_argument("tab", help="its title or id")
    commands.add_parser("clear", help="remove every tab, and the files in demo/tabs/")
    args = parser.parse_args()

    if args.command == "publish":
        publish(args.file, args.title, args.author)
    elif args.command == "list":
        for tab in tabs():
            print(f"{tab['id']}  {tab['title']}  (by {tab['author']}, revision {tab['revision']})")
    elif args.command == "delete":
        tab = find(args.tab)
        call("POST", "/tabs/delete", dict(id=tab["id"]))
        print(f"Deleted the tab {tab['title']!r}.")
    elif args.command == "clear":
        call("POST", "/tabs/reset", {})
        # The pages are meant to be written afresh each time, so the last run's files go too.
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs")
        for name in os.listdir(folder):
            if name.endswith(".html"):
                os.remove(os.path.join(folder, name))
        print("Deleted every tab, and the files in demo/tabs/.")


if __name__ == "__main__":
    main()
