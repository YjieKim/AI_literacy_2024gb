# 스트림릿 사용
import streamlit as st
import Python_ElectricCar_streamlit as ec
import pybasic as pb
import mProject_mbti as mp
import car_predict_up as cp

# 로그인 화면
st.sidebar.title("로그인")

userid = st.sidebar.text_input("아이디(ID) 입력", value='abc', max_chars=10) # value값을 기본으로 넣어둔 것.
userpw = st.sidebar.text_input("패스워드 입력", value='', type='password', max_chars=10)    # 값이 보이지 않게.

menu = ''
if userid == 'abc' and userpw == '1234':
    st.sidebar.title("환영합니다♥")

    menu = st.sidebar.radio("메뉴선택", ['파이썬 기초', '탐색적 분석: 전기자동차', '머신러닝: 중고차 가격 예측', '미니프로젝트: MBTI별 여가활동 추천'], index=None)
    # st.header(menu)

    if menu == '파이썬 기초':
        pb.basic()
    elif menu == '탐색적 분석: 전기자동차':
        ec.elec_exe()
    elif menu == '머신러닝: 중고차 가격 예측':
        #st.header("공사중...")
        #st.image('Hello.jpg')
        cp.aiml_main()
    elif menu == '미니프로젝트: MBTI별 여가활동 추천':
        mp.mbti_main()
