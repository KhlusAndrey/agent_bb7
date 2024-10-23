import datetime
from datetime import timezone

def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


def check_content(var):
    if var:
        try:
            var = var.content
            return var.content
        except:
            return var
    else:
        var