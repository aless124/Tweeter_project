from src import preprocessing, modeling
import os

def main():
    data_path = os.path.join('data', 'tweets.csv')
    metrics = modeling.train_evaluate(data_path)
    print(metrics)

if __name__ == '__main__':
    main()
