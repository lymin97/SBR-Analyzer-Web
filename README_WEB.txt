# SBR Analyzer Web Demo

Designed and developed by Young-Min Lee

이 폴더는 SBR Analyzer의 Streamlit 온라인 체험판입니다. 로컬 Windows 버전과 별도이며 Python 설치 MSIX나 BAT 파일이 필요하지 않습니다.

## 제공 기능

- 포함된 200 MBaud Differential 예제 채널로 즉시 분석
- Browse files 또는 드래그 앤 드롭으로 `.sNp` Touchstone 업로드
- Single-ended/Differential 포트 설정
- CTLE, FFE, DFE, FFE + DFE 설정
- Pulse Response와 Magnitude 결과 표시
- Cursor 및 EQ tap 표 표시
- PNG/CSV 개별 다운로드

온라인 버전에서는 업로드 파일의 실제 PC 경로를 읽지 않습니다. 파일 내용이 호스팅 서버의 임시 폴더로 전달되며 분석이 끝나면 해당 임시 폴더를 삭제합니다. 공개 배포에는 회사 기밀 또는 공개하면 안 되는 S-parameter 파일을 업로드하지 마십시오.

## 폴더 구성

```text
SBR_Analyzer_Web/
  streamlit_app.py
  sbr_analyzer.py
  sample_diff_channel_200Mbaud_minus6dB_at_nyquist.s4p
  requirements.txt
  runtime.txt
  .gitignore
  README_WEB.txt
```

## 로컬 시험 실행

Python 3.12 환경을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

명령 실행 후 브라우저에서 표시된 localhost 주소로 접속합니다. 종료할 때는 터미널에서 Ctrl+C를 누릅니다.

## GitHub에 업로드

1. GitHub에 로그인하고 New repository를 선택합니다.
2. 저장소 이름을 예를 들어 `SBR-Analyzer-Web`로 지정합니다.
3. 공개 체험판이면 Public을 선택합니다.
4. 이 폴더 안의 파일을 저장소 최상위 경로에 업로드합니다.
5. `streamlit_app.py`가 저장소 최상위에 있는지 확인합니다.
6. Commit changes를 눌러 업로드를 완료합니다.

로컬 버전의 `python-manager-*.msix`, BAT 파일, `last_settings.json`, `results`, `__pycache__`는 웹 저장소에 올릴 필요가 없습니다.

## Streamlit Community Cloud 배포

1. https://share.streamlit.io/ 에 접속하고 로그인합니다.
2. GitHub 계정을 연결하고 위 저장소 접근을 허용합니다.
3. Create app을 선택합니다.
4. Repository에서 업로드한 저장소를 선택합니다.
5. Branch는 보통 `main`을 선택합니다.
6. Main file path에 `streamlit_app.py`를 입력합니다.
7. 사용할 subdomain을 선택하거나 자동 생성을 사용합니다.
8. Deploy를 누릅니다.

배포 환경은 `requirements.txt`를 읽어 Streamlit, NumPy, SciPy, Matplotlib을 설치합니다. `runtime.txt`는 Python 3.12를 요청합니다. 배포가 끝나면 `https://...streamlit.app` 형태의 주소로 접속할 수 있습니다.

## 업데이트

GitHub 저장소에서 파일을 수정하거나 새 버전을 push하면 Streamlit이 앱을 다시 배포합니다. 의존성 변경은 `requirements.txt`에서 관리합니다.

## 문제 해결

- ModuleNotFoundError: `requirements.txt`가 저장소 최상위에 있는지 확인합니다.
- Sample file not found: 예제 `.s4p` 파일이 `streamlit_app.py`와 같은 폴더에 있는지 확인합니다.
- Invalid port: 업로드한 Touchstone의 실제 포트 수에 맞게 포트 번호를 지정합니다.
- Time record is too short: Pre/Post cursor 수를 줄이거나 더 낮은 주파수 간격과 충분한 sweep 범위를 가진 Touchstone을 사용합니다.
- 앱이 잠든 경우: 무료 호스팅은 일정 시간 미사용 시 sleep될 수 있으며 첫 접속이 느릴 수 있습니다.
- 업로드 실패: 호스팅 서비스의 파일 크기 제한과 네트워크 상태를 확인합니다.

## 공개 배포 주의사항

- 업로드 데이터는 서버에서 처리되므로 민감한 설계 파일에는 로컬 SBR Analyzer를 사용합니다.
- 이 체험판의 EQ 추천값은 참고용입니다.
- 실제 BER에는 noise, jitter, crosstalk, 회로 비선형성 및 구현 제약이 추가로 영향을 줍니다.
