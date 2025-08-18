# !!! non-functional yet
import subprocess
import time

start = time.perf_counter()
subprocess.run(["python", "outliers_sklearn_knn.py"], check=True)
end = time.perf_counter()

print(f"Runtime length: {end - start:.5f}s")
