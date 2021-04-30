from datetime import datetime

while True:
    inputs = input().split()

    start = datetime.strptime(inputs[0], "%H:%M:%S")
    end = datetime.strptime(inputs[1], "%H:%M:%S")

    diff = end - start
    print(diff.total_seconds())
    print("-------------------")
