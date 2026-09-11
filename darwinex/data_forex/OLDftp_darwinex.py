import os
import re
from datetime import datetime, timedelta
from ftplib import FTP

from tqdm import tqdm

FTP_SERVER = "tickdata.darwinex.com"
FTP_USER = "TheTrader99"
FTP_PASSWORD = "x8dhE1swCGt203"


class FtpDarwinex:

    def __init__(self, server=FTP_SERVER, username=FTP_USER, password=FTP_PASSWORD):
        self.ftp = FTP(server)
        self.ftp.login(username, password)

    def get_folder(self, symbol):
        self.ftp.cwd(f"/{symbol}/")

    def get_files(self):
        return self.ftp.nlst()

    def conditionated_files(self, start_date, end_date, side="BID"):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        valid_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, (end - start).days + 1)]

        files_list = self.get_files()

        date_filtered_files = [
            file for file in files_list
            if (match := re.search(r'\d{4}-\d{2}-\d{2}', file)) and match.group(0) in valid_dates
        ]

        gz_filtered_files = [file for file in date_filtered_files if file.endswith(".gz")]

        if side == "ASK":
            side_filtered_files = [file for file in gz_filtered_files if "ASK" in file]
        elif side == "BID":
            side_filtered_files = [file for file in gz_filtered_files if "BID" in file]
        else:
            side_filtered_files = [file for file in gz_filtered_files if "ASK" in file]
            bid_filtered_files = [file for file in gz_filtered_files if "BID" in file]
            side_filtered_files.extend(bid_filtered_files)

        return side_filtered_files

    def download_file(self, name, local_path):
        with open(local_path, "wb") as file:
            self.ftp.retrbinary(f"RETR {name}", file.write)

    def quit_connection(self):
        self.ftp.quit()


def get_asset_files(symbol, start_date, end_date, side, local_path):
    ftp = FtpDarwinex()
    ftp.get_folder(symbol)

    files_list = ftp.conditionated_files(start_date, end_date, side=side)

    for filename in tqdm(files_list, desc=symbol, leave=False):
        match = re.search(r'(\d{4})-(\d{2})-\d{2}', filename)
        year, month = match.group(1), match.group(2)
        local_dir = os.path.join(local_path, symbol, year, month)
        local_file = os.path.join(local_dir, filename)
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_file):
            continue

        try:
            ftp.download_file(filename, local_file)
        except Exception:
            pass

    ftp.quit_connection()