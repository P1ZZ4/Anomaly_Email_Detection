# 해당 코드는 jupyter lab에서 돌린 code이며, 편의상 py로 올려놓음 

# 본문 파싱 

## 사용한 라이브러리 
import email
import binascii
import quopri
import base64
import re
import os
import eml_parser
import quopri
from tld import get_tld
import csv
import base64
import sys
import urlextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from email.header import decode_header

## URL
# 본문에서 URL 추출하는 코드
def get_link(emllist, raw_body):
    url_extractor = urlextract.URLExtract()
    try:
        link = url_extractor.find_urls(raw_body)
        return link
    except Exception as e:
        print(emllist)
        print(e)
        return []
      
      

# boundary parsing
# Boundary를 기준으로 본문 섹션을 구분하기 위해서 사용
def parse_boundary(header):
    content_type_list = ''.join(header['content-type']).split(';')
    for item in content_type_list:
        if "boundary" in item:
            eml_boundary = item[item.find('boundary="') + 10:-1]

    return eml_boundary
  
  
# domain parsing
# 추출한 URL에서 도메인만 가져오는 코드
def parse_domain(eml_name, url_list):
    #sprint(str(type(url_list)))
    if url_list == []:
        return []
    elif url_list[0] == "":
        return []
    elif url_list[0] == url_list[-1]:
        try:
            tld = get_tld(url_list[0], as_object=True)
            return [tld.domain + '.' + tld.suffix]
        except Exception as e:
            print(eml_name, "domain error")
    elif url_list[0] != url_list[-1]:
        tld = []
        # url_list = [element for array in url_list for element in array]
        for url in url_list:
            try:
                tld_temp = get_tld(url, as_object=True)
                tld.append(tld_temp.domain + '.' + tld_temp.suffix)
            except Exception as e:
                tld += parse_domain(eml_name, url)
        return tld
    else:
        print(eml_name, "domain error - else")
        
        
        
        
        
# 본문 Decoding (plain)
# plain 형식의 본문을 디코딩하는 코드
# csv로 떨구다보니 개행같은 문자에서 행 구분이 잘 안되길래 빈칸으로 모두 치환함
# base64, quoted-printable이면 이거에 맞춰서 해독해준 다음에 디코딩했고,
# 그 외에는 그냥 디코딩만 하면 됨
def decode_plain(eml_name, raw_email, charset, encoding):
    if charset == None:
        charset = "utf-8"
    elif charset.upper() == "SHIFT_JIS":
        charset = "shift_jisx0213"
    elif charset.upper() == "CP-850":
        charset = "cp850"
    elif charset.upper() == "ISO-8859-":
        charset = "ISO-8859-1"

    try:
        raw_body = '\n\n'.join(raw_email.decode(charset, errors="ignore").split("\n\n")[1:])

        if encoding == None or encoding.lower() == "7bit" or encoding.lower() == "8bit" or encoding.lower() == "binary":
            plain_body = '\n\n'.join(raw_body.split("\n\n")[1:])
            plain_body = plain_body.replace("\n", " ")
            plain_body = plain_body.replace("\r", " ")
            plain_body = plain_body.replace(",", " ")
            return plain_body, get_link(eml_name, plain_body)

        elif encoding.lower() == "base64":
            try:
                plain_body = base64.b64decode(raw_body).decode(charset, errors="ignore")
            except:
                try:
                    plain_body = base64.b64decode(raw_body)
                    # print(html_body)
                except:
                    try:
                        plain_body = base64.b64decode(raw_body + "=").decode(charset, errors="ignore")
                    except:
                        plain_body = base64.b64decode(raw_body + "==").decode(charset, errors="ignore")
            plain_body = plain_body.replace("\n", " ")
            plain_body = plain_body.replace("\r", " ")
            plain_body = plain_body.replace(",", " ")
            return plain_body, get_link(eml_name, plain_body)

        elif encoding.lower() == "quoted-printable":
            raw_body = '\n\n'.join(raw_email.decode("utf-8", errors="ignore").split("\n\n")[1:])
            plain_body = quopri.decodestring(raw_body).decode(charset, errors="replace")
            plain_body = plain_body.replace("\n", " ")
            plain_body = plain_body.replace("\r", " ")
            plain_body = plain_body.replace(",", " ")
            return plain_body, get_link(eml_name, plain_body)

        else:
            raise decode_error()

    except Exception as e:
        print(eml_name + ": decode error")
        print(e)
        
        
# 본문 Decoding (HTML)
# html 형식의 본문을 디코딩하는 코드
# beautiful soup를 사용했음
# base64, quoted-printable은 먼저 디코딩해준 다음에 charset에 맞게 디코딩함
# 마찬가지로 정확한 행 구분을 위해 개행은 공란으로 치환함
# url도 beautiful soup 사용해서 가져왔음
def decode_html(eml_name, raw_email, charset, encoding):
    if charset == None:
        charset = "utf-8"
    elif charset.upper() == "SHIFT_JIS":
        charset = "shift_jisx0213"
    elif charset.upper() == "CP-850":
        charset = "cp850"
    elif charset.upper() == "ISO-8859-":
        charset = "ISO-8859-1"

    try:
        raw_body = '\n\n'.join(raw_email.decode(charset, errors="ignore").split("\n\n")[1:])

        if encoding == None or encoding.lower() == "7bit" or encoding == "8bit" or encoding == "binary":
            html_soup = BeautifulSoup(raw_body, "html.parser")
            html_text = html_soup.text
            html_href = []
            try:
                html_href_temp = html_soup.findAll('a', href=True)
                for i in html_href_temp:
                    html_href.append(i['href'])
            except:
                None
            html_text = html_text.replace("\n", " ")
            html_text = html_text.replace("\r", " ")
            html_text = html_text.replace(",", " ")
            return html_text, html_href

        elif encoding.lower() == "base64":
            try:
                html_body = base64.b64decode(raw_body).decode(charset, errors="ignore")
            except:
                try:
                    html_body = base64.b64decode(raw_body)
                    # print(html_body)
                except:
                    try:
                        html_body = base64.b64decode(raw_body + "=").decode(charset, errors="ignore")
                    except:
                        html_body = base64.b64decode(raw_body + "==").decode(charset, errors="ignore")
            html_soup = BeautifulSoup(html_body, "html.parser")
            html_text = html_soup.text
            html_href = []
            try:
                html_href_temp = html_soup.findAll('a', href=True)
                for i in html_href_temp:
                    html_href.append(i['href'])
            except:
                None
            html_text = html_text.replace("\n", " ")
            html_text = html_text.replace("\r", " ")
            html_text = html_text.replace(",", " ")
            return html_text, html_href

        elif encoding.lower() == "quoted-printable":
            raw_body = '\n\n'.join(raw_email.decode("utf-8", errors="ignore").split("\n\n")[1:])
            html_body = quopri.decodestring(raw_body).decode(charset, errors="ignore")
            # print(html_body)
            html_soup = BeautifulSoup(html_body, "html.parser")
            html_text = html_soup.text
            # print(html_text)
            html_href = []
            try:
                html_href_temp = html_soup.findAll('a', href=True)
                for i in html_href_temp:
                    html_href.append(i['href'])
            except:
                print(eml_name)
                print(e)
            html_text = html_text.replace("\n", " ")
            html_text = html_text.replace("\r", " ")
            html_text = html_text.replace(",", " ")
            return html_text, html_href

        else:
            raise decode_error()

    except Exception as e: 
        print(eml_name)
        print(e)
        
        
        
# 본문 Decoding (Multi)
# multi로 되어있어서  html이랑 plain 섞여있는 본문을 디코딩하기 위한 코드
# boundary로 구분해서 섹션별로 디코딩을 진행함
# 기재되어 있는 charset이라던가 encoding type을 알맞게 파싱하여 디코딩 함수로 전달
def decode_multi(eml_name, raw_email, charset, encoding, multi_boundary):
    if charset == None:
        charset = "utf-8"
    elif charset.upper() == "SHIFT_JIS":
        charset = "shift_jisx0213"
    elif charset.upper() == "CP-850":
        charset = "cp850"

    raw_body = b'\n\n'.join(raw_email.split(b"\n\n")[1:])
    multi_boundary = b"--" + multi_boundary.encode()
    multi_body = raw_body.split(multi_boundary)

    return_body = ""
    return_href = []

    for body in multi_body:
        if b"Content-Type: " in body:
            temp_body = ""
            temp_href = ""

            reg = re.compile('Content-Type: .*?\n')
            body_ct = reg.findall(body.decode())[0]
            # body_ct = ct[ct.find(' ')+1:ct.find(';')]

            reg = re.compile('Content-Transfer-Encoding: .*?\n')
            try:
                body_encoding = reg.findall(body.decode())[0]
                body_encoding = body_encoding.replace('Content-Transfer-Encoding: ', '')
                if body_encoding.find(';') > 0:
                    body_encoding = body_encoding[:body_encoding.find(';')]
                if body_encoding.find('\n') > 0:
                    body_encoding = body_encoding[:body_encoding.find('\n')]
                if body_encoding.find('>') > 0:
                    body_encoding = body_encoding[:body_encoding.find('>')]
                body_encoding = body_encoding.replace('"', '')
            except:
                body_encoding = None

            if "text/plain" in body_ct:
                # print("text/plain")
                reg = re.compile('charset.*\n')
                try:
                    body_charset = reg.findall(body.decode())[0][8:]
                    if body_charset.find(';') > 0:
                        body_charset = body_charset[:body_charset.find(';')]
                    if body_charset.find('\n') > 0:
                        body_charset = body_charset[:body_charset.find('\n')]
                    if body_charset.find('>') > 0:
                        body_charset = body_charset[:body_charset.find('>')]
                    body_charset = body_charset.replace('"', '')
                    body_charset = body_charset.replace('=', '')
                    body_charset = body_charset.replace('3D', '')
                except:
                    body_charset = "utf-8"

                temp_body, temp_href = decode_plain(eml_name, body, body_charset, body_encoding)
                return_body += " "
                return_body += temp_body
                return_href += temp_href

            elif "text/html" in body_ct:
                reg = re.compile('charset=.*?\n')
                try:
                    body_charset = reg.findall(body.decode())[0]
                    body_charset = body_charset.replace('charset=', '')
                    body_charset = body_charset.replace('3D', '')
                    if body_charset.find(';') > 0:
                        body_charset = body_charset[:body_charset.find(';')]
                    if body_charset.find('\n') > 0:
                        body_charset = body_charset[:body_charset.find('\n')]
                    if body_charset.find('>') > 0:
                        body_charset = body_charset[:body_charset.find('>')]
                    body_charset = body_charset.replace('"', '')
                    body_charset = body_charset.replace('=', '')
                except:
                    body_charset = "utf-8"

                if body_charset.find('"') > 0:
                    body_charset = body_charset[:body_charset.find('"')]
                if body_charset.find('=') > 0:
                    body_charset = body_charset[:body_charset.find('=')]
                if body_charset.find(' ') > 0:
                    body_charset = body_charset[:body_charset.find(' ')]
                temp_body, temp_href = decode_html(eml_name, body, body_charset, body_encoding)
                return_body += " "
                return_body += temp_body
                return_href += temp_href

    return return_body, return_href
  
  
  
# Main
# 헤더 파싱, 본문 디코딩, 결측치 채워서 csv에 넣는 작업
EmlDir = "./eml"
EmlLists = os.listdir(EmlDir)

if __name__ == "__main__":
    csv_fd = open('20201028_received_2.csv', 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_fd)
    row = ['eml', 'mime-version', 'date', 'x-accept-language', 'x-priority', 'reply-to', 'content-type',
           'content-transfer-encoding', 'rcpt_to', 'org_rcpt_to', 'x-helo', \
           'x-mail-from', 'message-id', 'to', 'from', 'x-mailer', 'x-received-ip', 'mail_from', 'received',
           'x-spam-type', 'body', 'href', 'href_domain', 'received_count', 'received', 'received-spf']

    
    csv_writer.writerows([row])

    EmlLists = os.listdir(EmlDir)
    count = 0
    for emllist in EmlLists:
        count += 1
        emlpath = "./eml/" + emllist
        if emllist == ".ipynb_checkpoints":
            continue
        elif emllist == ".idea":
            continue

        with open(emlpath, 'rb') as fhdl:
            try:
                raw_email = fhdl.read()
                ep = eml_parser.EmlParser()

                try:
                    parsed_eml = None
                    parsed_eml = ep.decode_email_bytes(raw_email)
                    # print(parsed_eml)
                except ValueError:
                    if "Message-ID: <[20\n" in raw_email.decode():
                        raw_email = '\n'.join(raw_email.decode().split("\n")[:-2]).encode()
                        parsed_eml = ep.decode_email_bytes(raw_email)
                    else:
                        # raw_email.
                        reg = re.compile('Date: .*?\n')
                        decode_email = raw_email.decode()
                        raw_date = reg.search(decode_email).group()
                        raw_email = (
                                decode_email[:decode_email.find(raw_date)] + "Date: Thu, 01 Jan 1970 00:00:00 +0000\n" + \
                                decode_email[decode_email.find(raw_date) + len(raw_date):]).encode()
                        parsed_eml = ep.decode_email_bytes(raw_email)

                except Exception as e:
                    print(emllist)
                    print(e)
                    continue

                try:
                    header_list = parsed_eml['header']['header']
                except:
                    print(emllist)
                    print(e)
                    continue

                # 데이터 초기화
                header_ct = None
                header_encoding = None
                header_charset = None

                ### body_text, body_href 초기화 ###
                html_soup = ""
                body_text = ""
                body_href = []
                body_href_temp = []
                domain = []
                received_count = 0
                received_last = ""

                #spf
                p = re.compile('Received-SPF\: (.*\n\t.*?\n\t.*|.*\n\t.*|.*\n)')
                received_spf_text = p.findall(raw_email.decode())
                if str(received_spf_text) == "[]":
                    received_spf_text = ""



                # 필요한 헤더 파싱
                if 'content-type' in header_list.keys():

                    header_ct = ''.join(parsed_eml['header']['header']['content-type'])
                    if "charset=" in header_ct:
                        header_charset = header_ct.split("charset=")[1]
                        if "\"" in header_charset:
                            header_charset = header_charset.replace('"', '')

                    if 'content-transfer-encoding' in parsed_eml['header']['header']:
                        header_encoding = ''.join(parsed_eml['header']['header']['content-transfer-encoding'])

                    ### 본문 및 링크 가져오는 코드 ###
                    if "text/html" in header_ct:
                        # print("text/html")
                        body_text, body_href = decode_html(emllist, raw_email, header_charset, header_encoding)
                        if body_href == []:
                            body_href += get_link(emllist, body_text)

                    elif "text/plain" in header_ct:
                        # print("text/palin")
                        if header_charset != None:
                            if header_charset.find(';') > 0:
                                header_charset = header_charset[:header_charset.find(';')]
                            if header_charset.find('\n') > 0:
                                header_charset = header_charset[:header_charset.find('\n')]
                            header_charset = header_charset.replace('"', '')
                            header_charset = header_charset.replace('=', '')
                            header_charset = header_charset.replace('3D', '')
                        elif header_charset == None:
                            None
                        body_text, body_href = decode_plain(emllist, raw_email, header_charset, header_encoding)

                    elif "multipart/alternative" in header_ct:
                        # print("multipart/alternative")
                        multi_boundary = parse_boundary(parsed_eml['header']['header'])
                        body_text, body_href = decode_multi(emllist, raw_email, header_charset, header_encoding,
                                                            multi_boundary)
                        if body_href == []:
                            body_href += get_link(emllist, body_text)

                    elif "multipart/mixed" in header_ct:
                        # print("multipart/mixed")
                        multi_boundary = parse_boundary(parsed_eml['header']['header'])
                        body_text, body_href = decode_multi(emllist, raw_email, header_charset, header_encoding,
                                                            multi_boundary)
                        if body_href == []:
                            body_href += get_link(emllist, body_text)

                    elif "multipart/related" in header_ct:
                        # print("multipart/related")
                        multi_boundary = parse_boundary(parsed_eml['header']['header'])
                        body_text, body_href = decode_multi(emllist, raw_email, header_charset, header_encoding,
                                                            multi_boundary)
                        if body_href == []:
                            body_href += get_link(emllist, body_text)

                    # ---------------------

                    if len(body_href) >= 1:
                        for url in body_href:
                            if r"http://" in url or r"https://" in url:
                                domain.append(urlparse(url).netloc)
                            else:
                                # print(url)
                                domain.append(urlparse(r"http://" + url).netloc)

                    # ---------------------
                else:
                    None

                #### csv에 저장하는 코드 ####
                try:
                    from_count = ReceviedFromCount(header_list['received'])
                except:
                    from_count = 0


                for item in row:
                    # print(item)

                    try:
                        if len(header_list[item]) == 1:
                            header_list[item] = header_list[item][0]
                        # print(item)
                        # print(header_list[item])
                    except:
                        header_list[item] = ""

                    if item == "reply-to" or item == "to" or item == "from":
                        if header_list[item].find("<") > 0:
                            header_list[item] = header_list[item][
                                                header_list[item].find("<") + 1:header_list[item].find(">")]
                    elif item == "received" and str(type(header_list[item])) == "<class 'list'>":
                        received_last = header_list[item][-1]
                    elif item == "date":
                        if header_list['date'] == "":
                            header_list['date'] = "Thu, 01 Jan 1970 00:00:00 +0000"
                    elif item == "x-received-ip":
                        if header_list['x-received-ip'] == "":
                            header_list['x-received-ip'] = "0.0.0.0"

                try:
                    if "<html>" in body_text:
                        html_soup = BeautifulSoup(body_text, "html.parser")
                        body_text = html_soup.text
                except:
                    print(emllist)
                    print("<html> error")

                csv_writer.writerow(
                    [emllist, header_list['mime-version'], header_list['date'], header_list['x-accept-language'], \
                     header_list['x-priority'], header_list['reply-to'], \
                     header_list['content-type'], header_list['content-transfer-encoding'], header_list['rcpt_to'], \
                     header_list['org_rcpt_to'], header_list['x-helo'], header_list['x-mail-from'], \
                     header_list['message-id'], header_list['to'], header_list['from'], \
                     header_list['x-mailer'], header_list['x-received-ip'], header_list['mail_from'], received_last, \
                     header_list['x-spam-type'], body_text, body_href, domain, from_count, header_list['received'], \
                     received_spf_text])

            except Exception as e:
                print(emllist)
                print(e)

    print("===========================")
    print(count)
    csv_fd.close()
    
    
# Exception
# 예외처리 하느라 사용했던 코드
# 이슈 다 처리해서 None으로 바꿨음
class decode_error(Exception):
    None

