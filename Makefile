PYTHON ?= .venv/bin/python
RUN ?= reproduce-01
V2_CONFIG ?=
V2_RUN_CONFIG = $(or $(V2_CONFIG),configs/v2.yaml)
V2_REPRODUCE_CONFIG = $(or $(V2_CONFIG),configs/v2_reproduce.yaml)
AUDIT_OUTPUT ?=
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 4

.PHONY: download verify test test-v1 reproduce report audit benchmark v2-preflight v2-reproduce v2-report v2-audit
download:
	$(PYTHON) scripts/download_data.py
verify:
	$(PYTHON) scripts/download_data.py --verify-only
test:
	$(PYTHON) -m pytest -q
test-v1:
	$(PYTHON) -m pytest -q --ignore-glob='tests/test_v2*'
reproduce:
	$(PYTHON) -m recsys_research.run --config configs/main.yaml --run-name $(RUN) --phase all
report:
	$(PYTHON) -m recsys_research.run --config configs/main.yaml --phase report
audit:
	$(PYTHON) scripts/audit_results.py --config configs/main.yaml --run-name $(RUN) $(if $(AUDIT_OUTPUT),--output "$(AUDIT_OUTPUT)")
benchmark:
	$(PYTHON) scripts/benchmark_scaling.py --config configs/main.yaml
v2-preflight:
	$(PYTHON) -m recsys_history_v2.run --config $(V2_RUN_CONFIG) --phase preflight
v2-reproduce:
	$(PYTHON) -m recsys_history_v2.run --config $(V2_REPRODUCE_CONFIG) --phase prepare
	$(PYTHON) scripts/profile_v2_gpu.py --config $(V2_REPRODUCE_CONFIG)
	$(PYTHON) -m recsys_history_v2.run --config $(V2_REPRODUCE_CONFIG) --phase search
	$(PYTHON) -m recsys_history_v2.run --config $(V2_REPRODUCE_CONFIG) --phase evaluate
	$(PYTHON) scripts/diagnose_v2_seeds.py --config $(V2_REPRODUCE_CONFIG)
	$(PYTHON) scripts/diagnose_v2_memory.py --config $(V2_REPRODUCE_CONFIG)
	$(PYTHON) -m recsys_history_v2.run --config $(V2_REPRODUCE_CONFIG) --phase report
	$(PYTHON) scripts/audit_v2_results.py --config $(V2_REPRODUCE_CONFIG)
v2-report:
	$(PYTHON) -m recsys_history_v2.run --config $(V2_RUN_CONFIG) --phase report
v2-audit:
	$(PYTHON) scripts/audit_v2_results.py --config $(V2_RUN_CONFIG) $(if $(AUDIT_OUTPUT),--output "$(AUDIT_OUTPUT)")
