"""The ``thinchat`` command line: manage the stored provider API keys.

A thin shell over ``thinchat.keys`` -- it parses ``argv``, calls the credential functions,
and renders for a human; no resolution logic lives here. Keys are identified by provider name
(``thinchat set claude``), matching the ``make_client`` roster, and stored via credbox in
``~/.config/thinchat/credentials.json`` (mode 0600). The full key value is never printed:
``set`` reads it with ``getpass`` (no echo), ``get`` shows it partially masked (only the
edges, or ``***`` when it is too short to show edges without revealing most of it), and
``list`` shows only which providers are set, never a value.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Callable, Sequence

from thinchat import __version__, keys
from thinchat.errors import ThinchatError

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the thinchat CLI. Returns the process exit code: 0 on success, 1 on a thinchat
    error (reported as a one-line message, not a traceback), 2 on a usage error. A missing
    key value on non-interactive stdin is a usage error, not a crash."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = _HANDLER_BY_COMMAND[args.command]
    try:
        return handler(args)
    except ThinchatError as err:
        print(f"thinchat: {err}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinchat", description="Manage the stored provider API keys.")
    parser.add_argument("--version", action="version", version=f"thinchat {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    set_parser = subcommands.add_parser("set", help="store a provider's API key")
    set_parser.add_argument("provider", help="the provider name (e.g. claude, openai, gemini)")

    get_parser = subcommands.add_parser("get", help="show a provider's resolved key, masked")
    get_parser.add_argument("provider", help="the provider name")

    subcommands.add_parser("list", help="list which providers have a stored key")

    unset_parser = subcommands.add_parser("unset", help="remove a provider's stored key")
    unset_parser.add_argument("provider", help="the provider name")

    return parser


def _cmd_set(args: argparse.Namespace) -> int:
    keys._stored_name(args.provider)   # reject an unknown/keyless provider before prompting
    try:
        entered = getpass.getpass(f"{args.provider} API key: ")
    except EOFError:
        print("thinchat: no key provided (stdin closed)", file=sys.stderr)
        return 2
    value = entered.strip()
    if not value:
        print("thinchat: no key provided", file=sys.stderr)
        return 2
    keys.set_api_key(args.provider, value=value)
    print(f"stored the API key for {args.provider}")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    key = keys.get_api_key(args.provider)
    if key is None:
        # Route the not-found line to stderr and exit non-zero, so `key=$(thinchat get X)` in a
        # script captures an empty stdout (and a testable exit code) rather than the literal
        # sentence "no key for X" mistaken for the key.
        print(f"thinchat: no key for {args.provider}", file=sys.stderr)
        return 1
    print(key)   # a Secret: str() masks it (edges only, or *** when too short to show edges)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    stored = set(keys.stored_providers())
    # Legend: `list` reflects the FILE store only -- an env var like OPENAI_API_KEY still resolves
    # via `get`/make_client but is deliberately not shown here, so "not set" never surprises a user
    # who has exported the key.
    print("stored keys (file store only; an env var like OPENAI_API_KEY also resolves but is not shown):")
    for provider in keys.ENV_BY_PROVIDER:
        print(f"  {provider:8} {'set' if provider in stored else 'not set'}")
    return 0


def _cmd_unset(args: argparse.Namespace) -> int:
    keys.unset_api_key(args.provider)
    print(f"removed the stored key for {args.provider}")
    return 0




# Command name -> handler. A dict dispatch (over argparse ``set_defaults``) keeps ``main``'s
# return type concrete for the type checker instead of an ``Any`` off the namespace.
_HANDLER_BY_COMMAND: dict[str, Callable[[argparse.Namespace], int]] = {
    "set": _cmd_set,
    "get": _cmd_get,
    "list": _cmd_list,
    "unset": _cmd_unset,
}
