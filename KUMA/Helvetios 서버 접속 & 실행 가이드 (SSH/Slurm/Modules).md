# KUMA/Helvetios 서버 접속 & 실행 가이드 (SSH/Slurm/Modules)

> 본 문서는 EPFL SCITAS 환경에서 **Helvetios(CPU 클러스터)**, **KUMA(GPU 클러스터: H100/L40S)**를 사용하는 연구자용 **SSH/Slurm/Modules** 실행 안내서입니다.  
> 전문용어는 English, 나머지는 한국어로 작성했습니다.

---

## 1) 클러스터 개요 (Helvetios vs KUMA)

- **Helvetios(예: `helvetios2`)**
  - 특징: **CPU-only**, `sinfo`의 `GRES`가 **(null)** 로 표시됨 → GPU 사용 불가
  - 용도: 문서/테스트/간단한 CPU 작업

- **KUMA(로그인: `kuma1`, 컴퓨트: `kh***` H100 / `kl***` L40S)**
  - 특징: **GPU 사용 가능**, `sinfo`에 `h100`, `l40s` 파티션과 `GRES=gpu:nvidia_h100|nvidia_l40s` 표시
  - 용도: GPU 학습/추론

---

## 2) SSH 접속

```bash
# Helvetios 로그인 (CPU 전용)
ssh <username>@helvetios2.epfl.ch

# KUMA 로그인 (GPU 사용 시 필수)
ssh <username>@kuma1.epfl.ch
````

**확인**:

```bash
hostname
sinfo -o "%P %G" | head -n 10
# Helvetios: PARTITION=standard*, GRES=(null) → GPU 없음
# KUMA: PARTITION=l40s/h100, GRES=gpu:nvidia_* → GPU 있음
```

---

## 3) Slurm 계정/권한 (Account/QOS) 핵심

* KUMA에서 **job 제출**은 프로젝트 **Account** 권한이 필요.
* 예: `-A dias` (허용), `-A master` (**disable**라면 제출 불가)
* KUMA 파티션 기본 QOS:

  * `h100`: `QoS=part_h100`
  * `l40s`: `QoS=part_l40s`
* **팁:** KUMA에서는 대개 `--qos` **생략**(기본 QOS 사용)이 안전.

**내 권한/한도 확인**:

```bash
sacctmgr show assoc where user=$USER format=Account,Partition,QOS,MaxJobsPU,MaxCPUsPU,MaxWall
sacctmgr show qos format=Name,MaxWall,MaxJobsPU,MaxTRESPU | grep -v disable
```

---

## 4) GPU 세션 열기 (srun/sbatch)

### 4.1 인터랙티브(권장: 짧게 확인)

```bash
# H100 노드 (권장)
srun -A dias -p h100 --gres=gpu:nvidia_h100:1 -c 2 --mem=16G -t 00:30:00 --pty bash

# L40S 노드 (대안)
srun -A dias -p l40s --gres=gpu:nvidia_l40s:1 -c 2 --mem=16G -t 00:30:00 --pty bash
```

성공 시 프롬프트가 `kh###`(H100) / `kl###`(L40S)로 변경.

### 4.2 배치(sbatch) 템플릿 (항상 통과 전략)

```bash
cat > run_min.sbatch <<'EOF'
#!/bin/bash
#SBATCH -A dias
#SBATCH -p h100            # 또는 l40s
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J gpu_check
#SBATCH -o slurm-%j.out

module load gcc/13.2.0
module load cuda/12.4.1

# 프로젝트 가상환경
source $HOME/Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua/.venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0))
PY
EOF

sbatch run_min.sbatch
squeue -u $USER
```

---

## 5) Modules & Python 가상환경

### 5.1 Modules 로드

```bash
module load gcc/13.2.0
module load cuda/12.4.1
```

### 5.2 가상환경 활성화

```bash
# 프로젝트 내 venv 예시
source $HOME/Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua/.venv/bin/activate
```

### 5.3 PyTorch (CUDA 12.4 권장 조합)

```bash
# Python 3.13 기준
pip install --force-reinstall \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0))
PY
```

> 로그인 노드(`kuma1`)에선 `avail=False`가 정상입니다. 반드시 **GPU 노드(예: `kh084`)**에서 확인하세요.

---

## 6) 프로젝트 실행 예시 (Make targets)

```bash
# 0) 저장소
git clone https://github.com/hyunjun1121/Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua.git
cd Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua

# 1) smoke (synthetic)
make smoke

# 2) real-data 락 (Wikitext-103)
make real-wt103-lock

# 3) 결과 집계 + 논문 자산
make lock
make paper

# 산출물 (예시)
# paper/RESULT_LOCK.md, paper/iclr_draft.md, paper/tables/*.tsv, paper/figs/*.png
```

---

## 7) 자주 발생하는 오류 & 해결

### 7.1 Slurm: `Invalid qos specification`

* 원인: KUMA에서 부적절한 `--qos` 지정, 또는 해당 계정에 권한 없음
* 해결: **`--qos` 제거**(기본 QOS 사용) 또는 `scontrol show partition <name>`로 `QoS=` 확인 후 해당 QOS 사용

### 7.2 Slurm: `QOSMaxCpuPerUserLimit` / `Job violates accounting/QOS policy`

* 원인: per-user 한도 초과(시간/CPU/TRES)
* 해결: 자원 축소(`-c 1`, `--mem 8G`, `-t 00:10:00`) → 안되면 **배치(sbatch)**로 제출

### 7.3 PyTorch: `cuda available: False`

* 원인: 로그인 노드에서 확인함 / 드라이버-휠 mismatch
* 해결: **GPU 노드에서 확인** / PyTorch를 **cu124 빌드**로 통일

### 7.4 RuntimeError (Attention dtype mismatch)

```
Input dtypes must be the same, got: input float, batch1: c10::BFloat16, batch2: c10::BFloat16
```

* 원인: `attn_mask=float32`, q/k/v=`bfloat16`
* 해결(권장): `src/train/build.py`에서 `self.attn(...)` 호출 직전 한 줄 추가
  `if attn_mask is not None and attn_mask.dtype != h.dtype: attn_mask = attn_mask.to(dtype=h.dtype)`
* 대안: `--precision fp32` 옵션 도입 또는 `model.float()` 강제

---

## 8) 체크리스트 (GPU 실험 전)

* [ ] 현재 노드가 **GPU 노드**인가? (`hostname`이 `kh***`/`kl***`)
* [ ] `-A dias`로 제출했는가? (KUMA에서 `master`는 보통 disable)
* [ ] `--gres=gpu:nvidia_h100:1` 또는 `--gres=gpu:nvidia_l40s:1` 정확한 타입 사용
* [ ] `module load gcc/13.2.0 && module load cuda/12.4.1`
* [ ] venv 활성화 + `torch==2.6.0+cu124` 계열 설치
* [ ] `make smoke` 통과 후 `make real-wt103-lock` → `make paper`

---

## 9) 슬럼 템플릿 (실험 파이프라인 배치)

```bash
cat > run_lock.sbatch <<'EOF'
#!/bin/bash
#SBATCH -A dias
#SBATCH -p h100
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J stobs_lock
#SBATCH -o slurm-%j.out

module load gcc/13.2.0
module load cuda/12.4.1
source $HOME/Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua/.venv/bin/activate

make real-wt103-lock
make paper
EOF

sbatch run_lock.sbatch
```

---

## 10) 문의 템플릿 (권한/QOS 문제 시)

```
Hello SCITAS,

I'm using the KUMA cluster for GPU jobs. My user is linked to the 'dias' project.
I get 'Invalid qos specification' on h100/l40s. Which QoS should I use for GPU jobs?
Partition configs show Default QoS as part_h100/part_l40s.

Thank you.
```

---

```
::contentReference[oaicite:0]{index=0}
```
