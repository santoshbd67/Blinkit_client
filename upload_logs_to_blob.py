
import os
from preProcUtilities import uploadFilesToBlobStore

log_dir = "./files"
print("tryu")
logs = [log for log in os.listdir(log_dir) if log.endswith(".log")]

# Uploading log files to blob
print("fgh",logs)
for log in logs:
    status, path = uploadFilesToBlobStore([os.path.join(log_dir,log)])
    print(status, path)
print("Uploadd logs Done!")
