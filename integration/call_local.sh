# Provider function for review-external.sh (qwen-2.5-localreview).
# Paste this function into review-external.sh alongside call_openai/call_google.
# See integration-guide.md for the full list of changes needed.

call_local() {
  local script="${LOCAL_REVIEW_SCRIPT}"
  local venv="${LOCAL_REVIEW_VENV}"
  local start_time
  start_time=$(date +%s)

  local raw_stderr
  raw_stderr=$(mktemp)
  local raw_output
  raw_output=$("${venv}/bin/python" "${script}" \
    --system "${SYSTEM_FILE}" --input "${USER_FILE}" 2>"${raw_stderr}") || {
    echo "[qwen-2.5-localreview] review script failed, skipping" >&2
    rm -f "${raw_stderr}"
    return 0
  }

  # Forward review.py's stderr (token counts) to our stderr.
  cat "${raw_stderr}" >&2
  rm -f "${raw_stderr}"

  if [[ -z "${raw_output}" ]]; then
    echo "[qwen-2.5-localreview] Empty response, skipping" >&2
    return 0
  fi

  local elapsed
  elapsed=$(( $(date +%s) - start_time ))
  echo "[qwen-2.5-localreview] ${elapsed}s -- \$0.00" >&2

  # Tag findings with provider (same pattern as openai/google providers).
  echo "${raw_output}" | while IFS= read -r line; do
    if [[ "${line}" =~ ^\[(BLOCK|WARN|NOTE)\] ]]; then
      echo "${line}" | sed -E 's/^\[([A-Z]+)\]/[\1] (qwen-2.5-localreview)/'
    elif [[ "${line}" == "No issues found." ]]; then
      echo "[qwen-2.5-localreview] No issues found." >&2
    fi
  done
}
