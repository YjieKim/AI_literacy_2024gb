# 진짜 최종(24.07.24)

import MBTI_food as mbti_fd
import mbti_workout_eunji as mbti_wo
import mbti_movie as mbti_mv
import streamlit as st

def mbti_main():
  
    # 시작할 때 안내 메시지 출력
    st.write(f"""
    어느새 벌써 7월 말! 이번 주말 어떤 계획을 가지고 계신가요?
    특별한 계획이 없으시다면, e조에서 당신만을 위한 제안 한 번 드려볼게요.

    MBTI에 맞게 당신만을 위해 준비했으니, 당신의 MBTI를 먼저 입력해주세요~
    """)

    st.divider()

    mbti_list = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]
    
    # user_mbti = ''
    user_mbti = st.selectbox(
        "당신의 MBTI는 무엇인가요?",
        mbti_list,
        index=None,
        placeholder="MBTI를 선택하세요.",
    )


    if user_mbti is not None:

        st.divider()

        recommend_food = mbti_fd.re_food(user_mbti)   # 음식 추천
        recommend_workout = mbti_wo.get_random_workout(user_mbti)   # 운동 추천
        recommend_movie, reason = mbti_mv.recommend_movie(user_mbti)    # 영화 추천

        st.markdown(f"""
당신의 MBTI는 {user_mbti}네요.

그럼 {user_mbti}인 당신만을 위한 추천 드립니다.


이번 주말에는 {recommend_food}를 드셔보세요.

(아주 많이) 덥고 습하지만… 운동도 잊으면 안 되죠.
{recommend_workout}도 한 번 해보세요~

마지막으로 시원한 맥주와 함께 '{recommend_movie}' 영화를 보며 주말 마무리를 해 보아요.
{reason}


그럼 행복한 주말 보내기를 바랍니다.

다음 주에도 심심하면 다시 방문해 주세요~
""")


if __name__=='__main__':
    mbti_main()