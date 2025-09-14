from learnandsavemodel import learn

def main():
    while True:
        print(f'какую модель использовать')
        print(f'1 - выход\n2 - начать обучение модели\n3 - использовать обученую модель')
        match input():
            case '1':
                return
            case '2':
                learn()
                continue
            case '3':
                print("Чтение из папок")
                print("/data/transactions")
                print("/data/transfers")
            case _:
                return
    while True
            
                
            


if __name__ == "__main__":
    main()