import learn as learn
import predict

def main():
    while True:
        print(f'Какую модель использовать:')
        print(f'1 - выход\n2 - начать обучение модели\n3 - использовать обученую модель')
        match input():
            case '1':
                return
            case '2':
                print("Учится на данных:\n/learn/transactions/*csv\n/learn/transfers/*.csv\n/learn/clients.csv")
                learn.learn()
                continue
            case '3':
                print("Чтение:\n/data/transactions/*csv\n/data/transfers/*.csv\n/data/clients.csv")
                predict.predict()
            case _:
                return
            
                
            


if __name__ == "__main__":
    main()