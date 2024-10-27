import pandas as pd
import openai
import gradio as gr
import asyncio

# OpenAI API 키 설정
openai.api_key = 'my_api_key'

# CSV 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    # CSV 파일 로드
    c = pd.read_csv(file_path, encoding='utf-8')
    
    # 불필요한 열 제거 및 폐과 상태 필터링
    c_data = c.drop(columns=['조사년도', '주야구분', '표준분류대계열', "표준분류중계열", "표준분류소계열", "설립구분"])
    c_data = c_data[c_data['대학구분'] != '대학원']
    c_data = c_data[c_data['학과상태'] != '폐지']

    return c_data

async def generate_response(prompt, filtered_data):
    # 필터링된 데이터가 데이터프레임일 경우 텍스트로 변환하여 포함
    if isinstance(filtered_data, pd.DataFrame):
        csv_text = filtered_data.to_string(index=False)
    else:
        csv_text = filtered_data  # filtered_data가 문자열일 경우 그대로 사용

    msg_history = [
        {"role": "system", "content": "당신은 진로, 진학 컨설턴트입니다. 사용자로부터 학교이름, 전공명, 필요한 정보의 키워드를 입력받으면 그에 맞게 대답해주세요."},
        {"role": "user", "content": f"다음 문서를 참고해주세요. {csv_text}"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=msg_history,
            max_tokens=700,
            temperature=0.7
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# CSV 데이터 필터링
def query_csv_data(query, c_data):
    # 사용자의 쿼리와 일치하는 데이터 필터링
    filtered_data = c_data[c_data.apply(lambda row: query.lower() in row.to_string().lower(), axis=1)]

    if filtered_data.empty:
        return "해당 조건에 맞는 정보가 없습니다."
    
    return filtered_data

async def chatbot_interface(user_input, file_path='data.csv'):
    # 전처리된 데이터 불러오기
    c_data = load_and_preprocess_data(file_path)
    
    # 쿼리에 맞는 데이터 필터링
    filtered_data = query_csv_data(user_input, c_data)
    
    return await generate_response(user_input, filtered_data)

async def main():
    iface = gr.Interface(
        fn=chatbot_interface,
        inputs="text",
        outputs="markdown",
        title="입시 정보 챗봇",
        description="질문을 입력하면 관련 입시 정보를 제공합니다."
    )
    await iface.launch(share=True)

if __name__ == "__main__":
    asyncio.run(main())
