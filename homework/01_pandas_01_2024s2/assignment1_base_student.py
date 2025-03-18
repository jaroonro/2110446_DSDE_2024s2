import pandas as pd

def main():
    file = input()
    func = input()

    df = pd.read_csv(file)

    if func == 'Q1':
        # Do something
        a = df.shape
    elif func == 'Q2':
        # Do something
        a = df.score.max()
    elif func == 'Q3':
        # Do something
        a = df[df.score>=80]['score'].count()
    else:
        # Do something
        a = "No Output"
    print(a)

if __name__ == "__main__":
    main()