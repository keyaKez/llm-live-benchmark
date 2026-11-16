#!/usr/bin/env bash
cd "$(dirname "$0")"

mkdir -p logs

# smoke test first: if it fails, stop before starting the long runs
echo "=== smoke test ==="
python main.py configs/smoke.yml 2>&1 | tee logs/smoke.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Smoke test failed (see logs/smoke.log). Stopping."
    exit 1
fi

# full experiments, smallest model first
configs=(
    configs/pythia-70m.yml
    configs/pythia-160m.yml
    configs/pythia-410m.yml
    configs/pythia-1b.yml
    configs/pythia-1.4b.yml
    configs/pythia-2.8b.yml
    configs/pythia-6.9b.yml
    configs/pythia-12b.yml
)

failed=()
for config in "${configs[@]}"; do
    name=$(basename "$config" .yml)
    echo "=== $name ==="
    python main.py "$config" 2>&1 | tee "logs/$name.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "$name failed (see logs/$name.log). Carrying on with the next one."
        failed+=("$name")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "All runs finished."
else
    echo "Runs that failed: ${failed[*]}"
    exit 1
fi
