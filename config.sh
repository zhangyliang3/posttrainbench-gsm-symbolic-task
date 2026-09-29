#!/usr/bin/env bash
# Source this file before running the DeepSeek-backed Harbor agent.
# Keep the API key in a separate, private file on the machine running Harbor.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Run this file with: source ./config.sh\n' >&2
  exit 1
fi

NOVIX_DEEPSEEK_KEY_FILE="${NOVIX_DEEPSEEK_KEY_FILE:-$HOME/.config/novix/deepseek_api_key}"
if [[ ! -f "$NOVIX_DEEPSEEK_KEY_FILE" || ! -r "$NOVIX_DEEPSEEK_KEY_FILE" ]]; then
  printf 'DeepSeek API key file is missing or unreadable: %s\n' "$NOVIX_DEEPSEEK_KEY_FILE" >&2
  return 1
fi

_novix_key="$(< "$NOVIX_DEEPSEEK_KEY_FILE")"
if [[ -z "$_novix_key" || "$_novix_key" == *$'\n'* ]]; then
  printf 'DeepSeek API key file must contain exactly one nonempty line.\n' >&2
  unset _novix_key
  return 1
fi

export ANTHROPIC_AUTH_TOKEN="$_novix_key"
export ANTHROPIC_API_KEY="$_novix_key"
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-flash"
unset _novix_key
