# Python 변환 스크립트 (MATLAB EELS background subtraction)

이 폴더는 기존 MATLAB 스크립트 3개를 Python으로 옮긴 버전입니다.

- `eels_fitting.py` → `EELS_fitting.m` 대응
- `eels_fit_analysis.py` → `EELS_fit_analysis.m` 대응
- `eels_subtracted_spectrum.py` → `EELS_subtracted_spectrum.m` 대응

## 설치

```bash
pip install numpy scipy matplotlib
```

## 1) 여러 exclusion 조건 비교 (`eels_fitting.py`)

```bash
python python/eels_fitting.py file_name.msa --model power2 --save-plot fitting.png
```

기본 exclusion 범위는 MATLAB 코드와 동일하게 `200:10:280` eV 입니다.

## 2) 단일 fit 분석 및 SNR 계산 (`eels_fit_analysis.py`)

```bash
python python/eels_fit_analysis.py file_name.msa \
  --model power2 \
  --exclude-above 280 \
  --save-prefix results/eels_fit \
  --save-plot analysis.png
```

생성 결과물:
- `*_residuals.csv`: `x_eV, counts, fitted, residuals`
- `*_params.json`: fit 계수, 공분산, `ik/ib/varib/h/snr`

## 3) MATLAB `.mat` 결과를 이용한 스펙트럼 플롯 (`eels_subtracted_spectrum.py`)

```bash
python python/eels_subtracted_spectrum.py file_name.msa \
  EELS_fit_xdata_file_name.mat \
  EELS_fit_ydata_residuals_file_name.mat \
  --x-var xdata2 --y-var residuals \
  --output-txt subtracted-spectrum.txt \
  --save-plot subtracted.png
```

`.mat` 파일 안에 변수가 1개뿐이면 `--x-var/--y-var`는 생략 가능합니다.
