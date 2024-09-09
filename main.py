# 스트림릿 사용
import streamlit as st
import electric_car as ec
import pybasic as pb
import mProject_mbti as mp
import car_predict as cp

# 로그인 화면
st.sidebar.title("로그인")

userid = st.sidebar.text_input("아이디(ID) 입력", value='abc', max_chars=10) # value값을 기본으로 넣어둔 것.
userpw = st.sidebar.text_input("패스워드 입력", value='1234', type='password', max_chars=10)    # 값이 보이지 않게.

menu = ''
if userid == 'abc' and userpw == '1234':
    st.sidebar.subheader("YJ의 포트폴리오")
    st.sidebar.divider()
    
    menu = st.sidebar.radio("메뉴선택", ['소개', '파이썬 기초', '탐색적 분석: 전기자동차', '머신러닝: 중고차 가격 예측', '미니프로젝트: MBTI별 여가활동 추천'])
    st.header(menu)

    if menu == '소개':
        st.write("YJ의 포트폴리오에 오신 것을 환영합니다.")
        st.write("여기에 있는 내용은 2024년 강북여성새로일하기센터 AI(인공지능) 리터러시 과정 중 일부입니다.")
        st.write("학습을 위해 임으로 변경한 내용이 있을 수 있으며, 교육 중 학습한 내용이 추가될 수 있습니다.")
    elif menu == '파이썬 기초':
        pb.basic()
    elif menu == '탐색적 분석: 전기자동차':
        ec.elec_exe()
    elif menu == '머신러닝: 중고차 가격 예측':
        cp.aiml_main()
    elif menu == '미니프로젝트: MBTI별 여가활동 추천':
        mp.mbti_main()
