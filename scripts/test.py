import time
import datetime
import cv2
import numpy as np
import pandas as pd
import openpyxl


def foo(t):
    print(f'time: {t}')


excel_path = '../data/test.xlsx'
old_time = time.time()
begin_time = old_time
lst = []
while True:
    new_time = time.time()
    if new_time - begin_time > 10:
        break
    if new_time - old_time > 2:
        right_now = str(datetime.datetime.now())
        lst.append(right_now)
        foo(right_now)
        old_time = new_time

df = pd.DataFrame(np.array(lst).reshape(-1, 1), columns=['time'])
df.to_excel(excel_path, sheet_name='test1', index=False, header=True)

