import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 讀取 CSV 檔案
try:
    df = pd.read_csv(r"D:\Old.D_\大學\專題\競賽資料\Code\data_xml\csv_file\output.csv")
    print("CSV 檔案讀取成功。")

    # 數據預處理
    if 'PositionLon' in df.columns and 'PositionLat' in df.columns and 'VDID' in df.columns:
        # 提取經緯度數據
        coordinates = df[['PositionLon', 'PositionLat']].values.astype(float)

        # 孤立森林 (Isolation Forest)
        isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=0)
        isolation_forest.fit(coordinates)
        outliers = isolation_forest.predict(coordinates)

        # 找出異常設備
        anomalous_devices = df[outliers == -1]

        # 輸出結果
        if not anomalous_devices.empty:
            print("\n偵測到以下異常設備：")
            for index, row in anomalous_devices.iterrows():
                print(f"設備 ID: {row['VDID']}, 經度: {row['PositionLon']}, 緯度: {row['PositionLat']}")
        else:
            print("\n未偵測到異常設備。")

    else:
        print("\nCSV 檔案中缺少 'PositionLon'、'PositionLat' 或 'VDID' 列。請檢查數據。")

except FileNotFoundError:
    print("找不到 CSV 檔案，請檢查路徑是否正確。")
except Exception as e:
    print(f"讀取 CSV 檔案時發生錯誤：{e}")