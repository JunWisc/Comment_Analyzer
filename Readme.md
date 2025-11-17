# Comment Analyzer 사용 방법

## 다운

`pip install requests google-genai`  
`pip install beautifulsoup4`  

## 실행

`python comment.py`  

## 방법

실행 후, 

1. 한국어 (K) 혹은 영어 (E) 작성 (소문자 대문자 상관 없음)  
2. url 입력  
3. 분석 진행 (이때, 뭔가 멈춘것 같더라도 엄춘게 아니니 기다린다!)  
4. 첫번째 시도라면 종료 (T) 하면 바로 종료. 비교하고 싶으면 다시 K or E 후 URL 넣기.  
5. 두번째 이상 (즉, 비교할 url 이 두개 이상) 이고, T를 하면 서로 분석 시작!  

만약, 어디서부터 시작인지 모르겠으면 여기! 를 검색한다.  
만약, 코멘트 개수를 수정하고 싶으면 MAX_COMMENTS = 의 숫자를 변경한다  

## 에러

500 INTERNAL. {'error': {'code': 500, 'message': 'An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting', 'status': 'INTERNAL'}}  
URL을 다시 입력해주세요. 라고 뜨면, 코드 문제가 아니니 다시 url을 검색한다!  
