from learnandsavemodel import learn

def main():
    while True:
        print(f'q - выход\nn - начать обучение модели\nl - загрузить\np - предсказать')
        match input():
            case 'q':
                return
            case 'n':
                learn()
            case 'l':
                
            case 'p':
                
            case _:
                continue
            
                
            


if __name__ == "__main__":
    main()