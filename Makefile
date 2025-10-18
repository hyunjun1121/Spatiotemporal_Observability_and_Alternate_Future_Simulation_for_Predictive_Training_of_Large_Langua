.PHONY: smoke branch-test torchrun test ablate replay real-wt103 real-c4small real-wt103-lock real-c4small-lock lock paper docker

smoke:
	python -m src.main --mode baseline --dataset synthetic --steps 400 --batch_size 8 --seq_len 128

branch-test:
	python -m src.main --mode branch-test --dataset synthetic --steps 200 --batch_size 8 --seq_len 128

torchrun:
	torchrun --nproc_per_node=2 -m src.main --mode baseline --dataset synthetic --steps 800 --batch_size 8 --seq_len 128

test:
	pytest -q

ablate:
	bash scripts/run_ablation.sh

replay:
	bash scripts/run_replay.sh

real-wt103:
	bash scripts/run_replicates.sh --real_data wikitext --baseline fixedlr --method C --config assets/experiments/wikitext_rc1.yaml

real-c4small:
	bash scripts/run_replicates.sh --real_data c4 --baseline fixedlr --method C --config assets/experiments/c4small_rc1.yaml

real-wt103-lock:
	bash scripts/run_replicates.sh --real_data wikitext --baseline fixedlr,hypergrad,zclip,spamlite,pbtlite --method A,B,C,D --config assets/experiments/wikitext_rc1.yaml

real-c4small-lock:
	bash scripts/run_replicates.sh --real_data c4 --baseline fixedlr,hypergrad,zclip,spamlite,pbtlite --method A,B,C,D --config assets/experiments/c4small_rc1.yaml

lock: real-wt103-lock real-c4small-lock
	python -m src.eval.aggregate --runs_root runs --paper_dir paper --groupby experiment,baseline,method --lock_json lock.json

paper:
	python -m scripts.gen_paper_assets

docker:
	docker build -t stobs:rc1 .
