import schedule
import time
from config import DEADLINE
from datetime import datetime
from batch_process import batch_process

def start_batch_processing():
    print(f"Starting batch processing at {datetime.now()}...")
    batch_process()

schedule.every().day.at(DEADLINE.split()[1]).do(start_batch_processing)

print(f"Waiting for the evaluation to start at {DEADLINE}...")
while True:
    schedule.run_pending()
    time.sleep(60)