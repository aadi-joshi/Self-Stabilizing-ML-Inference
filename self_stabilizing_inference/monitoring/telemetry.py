import pandas as pd
import time

class TelemetryLogger:
    def __init__(self):
        self.df = pd.DataFrame()

    def log(self, record):
        record["timestamp"] = time.time()
        self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
