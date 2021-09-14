import smtplib
import time
from email.header import Header
from email.mime.text import MIMEText

# 第三方 SMTP 服务
mail_host="smtp.shanghaitech.edu.cn"  # set smtp server
mail_user="zhangchg@shanghaitech.edu.cn"    # username
mail_pass="I3e436"   #passwd


sender = mail_user
receivers = mail_user  # target mail address
cmd = "nvidia-smi -a | grep \"Gpu\""

# vim remote_run.py
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--m', default=0, type=int, metavar='N')
args = parser.parse_args()

if args.m == 0:
    g_list = ['g19','g20','g13','g14','g15','g26','g27','g28']
    g_list_flag = [True]*7
else:
    g_list = ['node10','node16','node17','node32','node34'] # ,'node19'
    g_list_flag = [True]*6
while True:
    for g_i, g in enumerate(g_list):
        if not g_list_flag[g_i]:
            continue
        if args.m == 0:
            client.connect(g, username='root',password="251314",port=21097)
        else:
            client.connect(g, username='root',password="plus",port=22586)
        stdin, stdout, stderr = client.exec_command(cmd)
        # print(stdout.readlines())

        flag=0
        text = ""
        for i in stdout.readlines():
            if i.find(": 0 %")!=-1:
                flag +=1
            text += i
        # print(text)
        if flag >=4:
            g_list_flag[g_i] = False
            try:
                smtpObj = smtplib.SMTP()
                smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
                smtpObj.login(mail_user, mail_pass)

                message = MIMEText(text, 'plain')
                # message = MIMEMultipart("Gpu 空闲！")
                message['From'] = Header(g, 'utf-8')
                message['To'] =  Header("zhangchg")
                subject = g+'idle'
                message['Subject'] = Header(subject)
                # message.attach(message_text)
                print(message.as_string())

                smtpObj.sendmail(sender, receivers, message.as_string())
                print("邮件发送成功")
            except smtplib.SMTPException:
                print("Error: 无法发送邮件")
    if g_list_flag.count(True):
        time.sleep(60)
        # print("sleep")
    else:
        break
