# 해당 코드는 jupyter lab에서 돌린 code이며, 편의상 py로 올려놓음 

# 결측치 채우기
## 본문 파싱 > 메인에 있는 코드 (일부 발췌)
elif item == "date":
   if header_list['date'] == "":
      header_list['date'] = "Thu, 01 Jan 1970 00:00:00 +0000"
elif item == "x-received-ip":
   if header_list['x-received-ip'] == "":
      header_list['x-received-ip'] = "0.0.0.0"
