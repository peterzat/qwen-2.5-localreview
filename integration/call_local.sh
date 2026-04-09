# Provider function for review-external.sh (qwen-2.5-localreview).
# Paste this function into review-external.sh alongside call_openai/call_google.
# See integration-guide.md for the full list of changes needed.

call_local() {
  local script="${LOCAL_REVIEW_SCRIPT}"
  local venv="${LOCAL_REVIEW_VENV}"

  local raw_stderr
  raw_stderr=$(mktemp)
  local raw_output
  raw_output=$(timeout "${TIMEOUT}" "${venv}/bin/python" "${script}" \
    --system "${SYSTEM_FILE}" --input "${USER_FILE}" 2>"${raw_stderr}") || {
    # Forward captured stderr before reporting failure (aids diagnosis).
    cat "${raw_stderr}" >&2
    echo "[qwen] review script failed, skipping" >&2
    rm -f "${raw_stderr}"
    return 0
  }

  # Forward review.py's stderr (token counts + timing) to our stderr.
  cat "${raw_stderr}" >&2
  rm -f "${raw_stderr}"

  if [[ -z "${raw_output}" ]]; then
    echo "[qwen] Empty response, skipping" >&2
    return 0
  fi

  # Tag findings with provider (same pattern as openai/google providers).
  # Use printf + here-string to avoid echo mangling backslashes.
  while IFS= read -r line; do
    if [[ "${line}" =~ ^\[(BLOCK|WARN|NOTE)\] ]]; then
      printf '%s\n' "${line}" | sed -E 's/^\[([A-Z]+)\]/[\1] (qwen)/'
    elif [[ "${line}" == "No issues found." ]]; then
      echo "[qwen] No issues found." >&2
    fi
  done <<< "${raw_output}"
}
