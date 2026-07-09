.PHONY: run compare privacy-utility report check clean-catboost all-report

run:
	python run_all.py

compare:
	python compare_all_results.py

privacy-utility:
	python privacy_utility_summary.py

report:
	python generate_final_report.py

check:
	python check_reproducibility.py

clean-catboost:
	rm -rf catboost_info

all-report:
	python compare_all_results.py
	python privacy_utility_summary.py
	python generate_final_report.py
	python check_reproducibility.py
