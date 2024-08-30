import streamlit as st
import random

# 구구단 함수
def gugudan():
    dan = st.number_input("수를 입력하세요(1 이상): ", value=1)
    # dan = int(st.text_input("수를 입력하세요(1 이상): ", value='1'))

    if dan >= 1:

        for i in range(1,10):
            st.write(f"{dan} * {i} = {dan*i}")

# 음식추천
def recommand_food():
    c_food = ['자장면','짬뽕','탕수육','팔보채','유산슬']
    k_food = ['비빔밥','갈비탕','잔치국수','김치찌개']

    menu = st.radio("음식추천", ['중식', '한식'], index=None)

    if menu == '중식':
        st.write(f"오늘의 중식 추천메뉴 : {random.choice(c_food)}")
    elif menu == '한식':
        st.write(f"오늘의 한식 추천메뉴 : {random.choice(k_food)}")
    else:
        st.write("선택하세요.")

# 화면 구성 : 탭
def basic():
    tab1, tab2 = st.tabs(["구구단", "음식추천"])

    with tab1:
        st.subheader("구구단 프로그램")
        gugudan()
    with tab2:
        st.subheader("음식추천 프로그램")
        recommand_food()