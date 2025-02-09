import pandas as pd
import numpy as np
import subprocess
from datetime import datetime

# 嘗試導入 LangChain，如果失敗則安裝
try:
    import langchain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    print("LangChain 已安裝。")
except ImportError:
    print("LangChain 未安裝，正在安裝...")
    try:
        subprocess.check_call(["pip", "install", "langchain"])
        import langchain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        print("LangChain 安裝成功。")
    except Exception as e:
        print(f"LangChain 安裝失敗：{e}")
        exit()  # 安裝失敗則退出

# 讀取 CSV 檔案
try:
    df = pd.read_csv(r"D:\Old.D_\大學\專題\競賽資料\Code\data_xml\csv_file\output.csv")
    print("CSV 檔案讀取成功。")

    # 確認欄位格式
    print("\n欄位格式：")
    print(df.dtypes)

    # 移除缺失值
    print("\n移除缺失值...")
    df = df.dropna(subset=['PositionLat', 'PositionLon'])
    print("已移除缺失值。")

    # 檢查 VDID 是否重複
    print("\n檢查 VDID 是否重複...")
    duplicate_vdids = df[df.duplicated(subset=['VDID'], keep=False)]
    if not duplicate_vdids.empty:
        print("偵測到以下重複的 VDID：")
        print(duplicate_vdids[['VDID', 'PositionLat', 'PositionLon']])
    else:
        print("未偵測到重複的 VDID。")

    # 檢查 VDID 是否跳號或遺失
    print("\n檢查 VDID 是否跳號或遺失...")
    # 提取 VDID 中的數字部分
    df['VDID_Number'] = df['VDID'].str.extract(r'(\d+)').astype(int)
    # 按照 VDID_Number 排序
    df = df.sort_values(by='VDID_Number')
    # 檢查是否有跳號
    missing_vdids = []
    for i in range(1, df['VDID_Number'].max() + 1):
        if i not in df['VDID_Number'].values:
            missing_vdids.append(i)
    if missing_vdids:
        print("偵測到以下跳號或遺失的 VDID：")
        print(missing_vdids)
    else:
        print("未偵測到跳號或遺失的 VDID。")

    # 準備 LangChain
    # 需要 OpenAI API 金鑰
    try:
        with open("API_KEY.txt", "r") as f:
            openai_api_key = f.read().strip()
    except FileNotFoundError:
        print("找不到 API_KEY.txt 檔案，請確認檔案是否存在。")
        exit()

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # 設計 PromptTemplate
    template = """
    您是一位專業的數據分析師，請根據以下資訊生成一份關於國道測速設備異常分析的報告，報告格式為 Markdown。

    資料來源：{csv_file_path}
    分析日期：{analysis_date}

    資料預處理結果：
    欄位格式：{column_types}
    缺失值移除數量：{missing_value_count}

    異常偵測結果：
    VDID 重複情況：{duplicate_vdid_info}
    VDID 跳號或遺失情況：{missing_vdid_info}

    請根據以上資訊，撰寫一份詳細且專業的分析報告，並提出相關建議。
    """

    prompt = PromptTemplate(
        input_variables=["csv_file_path", "analysis_date", "column_types", "missing_value_count", "duplicate_vdid_info", "missing_vdid_info"],
        template=template,
    )

    # 準備報告內容
    csv_file_path = r"D:\Old.D_\大學\專題\競賽資料\Code\data_xml\csv_file\output.csv"
    analysis_date = datetime.now().strftime("%Y-%m-%d")
    column_types = str(df.dtypes)
    missing_value_count = df[['PositionLat', 'PositionLon']].isnull().sum().sum()
    duplicate_vdid_info = "偵測到以下重複的 VDID：\n" + str(duplicate_vdids[['VDID', 'PositionLat', 'PositionLon']]) if not duplicate_vdids.empty else "未偵測到重複的 VDID。"
    missing_vdid_info = "偵測到以下跳號或遺失的 VDID：\n" + str(missing_vdids) if missing_vdids else "未偵測到跳號或遺失的 VDID。"

    # 建立 LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # 執行 LLMChain
    report = chain.run(csv_file_path=csv_file_path, analysis_date=analysis_date, column_types=column_types, missing_value_count=missing_value_count, duplicate_vdid_info=duplicate_vdid_info, missing_vdid_info=missing_vdid_info)

    # 輸出報告
    print("\n報告內容：")
    print(report)

except FileNotFoundError:
    print("找不到 CSV 檔案，請檢查路徑是否正確。")
except Exception as e:
    print(f"讀取 CSV 檔案時發生錯誤：{e}")